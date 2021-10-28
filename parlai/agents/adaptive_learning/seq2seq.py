import torch
import torch.nn.functional as F
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from .criterions import LabelSmoothing, CrossEntropyLabelSmoothing
from .helper import build_loss_desc, build_prob_desc, compute_batch_loss
import torch.nn as nn
from scipy.spatial import distance


class AdaSeq2seqAgent(Seq2seqAgent):

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        # self.prev_mean_input_emb = None
        self.prev_batch_input = None
        self.margin = nn.Parameter(torch.Tensor([opt['margin']]))
        self.margin.requires_grad = False
        self.margin_rate = opt['margin_rate']

        if torch.cuda.is_available():
            self.margin = self.margin.cuda()

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
            self.batch_criterion = LabelSmoothing(
                len(self.dict), self.NULL_IDX)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)
            self.batch_criterion = CrossEntropyLabelSmoothing(
                len(self.dict), self.NULL_IDX)

        if self.use_cuda:
            self.criterion.cuda()
            self.batch_criterion.cuda()

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        #self.pretrain_weight
        self.zero_grad()

        try:
            loss, margin_loss, model_output, cur_batch_input_emb = self.compute_loss(batch, return_output=True)
            batch_loss = compute_batch_loss(model_output, batch, self.batch_criterion, self.NULL_IDX)
            self.metrics['loss'] += loss.item()
            self.backward(loss)
            self.update_params()
            return loss, model_output, batch_loss, margin_loss, cur_batch_input_emb
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        batch_size = len(observations)

        # check if there are any labels available, if so we will train on them
        is_training = any('labels' in obs for obs in observations)

        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID(), 'is_training': is_training}
                       for _ in range(batch_size)]

        # create a batch from the vectors and pad to same length
        batch = self.batchify(observations)

        if is_training:
            train_return = self.train_step(batch)
            if train_return is not None:
                #            return loss, model_output, batch_loss, margin_loss, mean_input_embed
                _, model_output, batch_loss, margin_loss, cur_batch_input_emb = train_return
                # print('batch_act mean_input_embed', mean_input_embed)
                scores, *_ = model_output
                scores = scores.detach()
                batch_loss = batch_loss.detach()
            else:
                batch_loss = None
                scores = None
                margin_loss = None

            self.replies['batch_reply'] = None
            # TODO: add more model state or training state for sampling the next batch
            #       (learning to teach)
            train_report = self.report()
            loss_desc = build_loss_desc(batch_loss, self.use_cuda)
            prob_desc = build_prob_desc(scores, batch.label_vec, self.use_cuda, self.NULL_IDX)
            for idx, reply in enumerate(batch_reply):
                reply['train_step'] = self._number_training_updates
                reply['train_report'] = train_report
                reply['loss_desc'] = loss_desc
                reply['margin_loss'] = margin_loss
                reply['prob_desc'] = prob_desc
                reply['cur_batch_input_emb'] = cur_batch_input_emb
            return batch_reply
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back graidients.
                # noinspection PyTypeChecker
                eval_output = self.eval_step(batch)
                if eval_output is not None:
                    output = eval_output[0]
                    label_text = eval_output[1]
                    context = eval_output[2]
                    if label_text is not None:
                        # noinspection PyTypeChecker
                        self._eval_embedding_metrics(output, label_text, context)
                        # noinspection PyTypeChecker
                        self._eval_distinct_metrics(output, label_text)
                        self._eval_entropy_metrics(output, label_text)
                else:
                    output = None

            if output is None:
                self.replies['batch_reply'] = None
                return batch_reply
            else:
                self.match_batch(batch_reply, batch.valid_indices, output)
                self.replies['batch_reply'] = batch_reply
                self._save_history(observations, batch_reply)  # save model predictions
                return batch_reply

    def _model_input(self, batch):
        """
        Creates the input (x) value for the model. Must return a tuple.
        This will be passed directly into the model via *args, i.e.,

        >>> model(*_model_input(batch))

        This is intentionally overridable so that richer models can pass the
        additional inputs.
        """
        return (batch.text_vec, )

    def compute_loss(self, batch, return_output=False):
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        # if self.prev_mean_input_emb is not None:
        #     model_output = self.model(*self._model_input(batch), ys=batch.label_vec, prev_emb = self.prev_mean_input_emb.detach())
        # else:
        cur_batch_input = self._model_input(batch)
        model_output = self.model(*cur_batch_input, ys=batch.label_vec)

        scores, preds, encoder_states = model_output
        score_view = scores.view(-1, scores.size(-1))
        generation_loss = self.criterion(score_view, batch.label_vec.view(-1))

        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == preds) * notnull).sum().item()

        loss = generation_loss/target_tokens  # average loss per token
        # print('self.prev_batch_input', self.prev_batch_input)
        # print('self.cur_batch_input', cur_batch_input[0])
        cur_batch_input_emb = self.model.pretrain_embedding(cur_batch_input[0])
        if self.prev_batch_input is not None and len(batch.text_vec) == self.opt['batchsize']:
            prev_batch_input_emb = self.model.pretrain_embedding(self.prev_batch_input)
            # prev_batch_input_emb_mean = torch.sum(torch.mean(prev_batch_input_emb, 0),0)
            # cur_batch_input_emb_mean = torch.sum(torch.mean(cur_batch_input_emb, 0),0)
            # margin_loss = -1 * distance.cosine(prev_batch_input_emb_mean.cpu(), cur_batch_input_emb_mean.cpu())
            margin_loss = -F.cosine_similarity(prev_batch_input_emb, cur_batch_input_emb).abs().mean()
            loss = self.margin_rate * margin_loss + (1 - self.margin_rate) * generation_loss
        else:
            # loss = generation_loss
            margin_loss = -1

        # print('compute_loss prev_emb', prev_emb)
        # print('compute_loss mean_input_embed', mean_input_embed)

        # if len(batch.text_vec) == self.opt['batchsize']:
        #      self.prev_mean_input_emb = mean_input_embed
        if len(batch.text_vec) == self.opt['batchsize']:
            self.prev_batch_input = cur_batch_input[0] #self._model_input(batch)[0]
        # save loss to metrics
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens

        if return_output:
            return (loss, margin_loss, model_output, cur_batch_input_emb)
        else:
            return (loss, margin_loss)