#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#
from abc import ABC
from typing import Any
import torch.nn as nn
from my_utils import *


# Multiple Futures Prediction Network
class mfpNet(nn.Module, ABC):
    def __init__(self, args) -> None:
        super(mfpNet, self).__init__()
        # type: ignore
        self.use_cuda = args['use_cuda']
        self.encoder_size = args['encoder_size']  # encoder latent layer size
        self.decoder_size = args['decoder_size']  # decoder latent layer size
        # number of original future samples / factor subsample in time
        self.out_length = args['fut_len_orig_hz'] // args['subsampling']

        self.dyn_embedding_size = args['dyn_embedding_size']  # dynamic embedding size
        self.input_embedding_size = args['input_embedding_size']  # input embedding size

        self.nbr_atten_embedding_size = args['nbr_atten_embedding_size']  # neighborhood attention embedding size
        self.st_enc_hist_size = self.nbr_atten_embedding_size

        self.st_enc_pos_size = args['dec_nbr_enc_size']  # decoder neighbors encode size
        self.use_gru = args['use_gru']  # GRUs instead of LSTMs
        self.bi_direc = args['bi_direc']  # bidirectional
        self.use_context = args['use_context']  # use contextual image as additional input
        self.modes = args['modes']  # how many latent modes
        self.use_forcing = args['use_forcing']  # 0: Teacher forcing. 1:classmates forcing.

        self.hidden_fac = 2 if args['use_gru'] else 1
        self.bi_direc_fac = 2 if args['bi_direc'] else 1
        self.dec_fac = 2 if args['bi_direc'] else 1

        # call function for initialization
        self.init_rbf_state_enc(in_dim=self.encoder_size * self.hidden_fac)
        self.posi_enc_dim = self.st_enc_pos_size
        self.posi_enc_ego_dim = 2

        # Input embedding layer, typically we choose 32
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoding RNN, encoder_size is chosen to be 16
        # see help document for more detail
        if not self.use_gru:
            self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        else:
            # use two layers for GRU
            self.num_layers = 2
            self.enc_lstm = torch.nn.GRU(self.input_embedding_size, self.encoder_size,
                                         num_layers=self.num_layers, bidirectional=False)

        # Dynamics embeddings, dyn_embedding_size is 32
        # num_layers=2 means two parallel GRU units with different hidden state and cell state
        self.dyn_emb = torch.nn.Linear(self.encoder_size * self.hidden_fac, self.dyn_embedding_size)

        # Define feature from semantic map
        context_feat_size = 64 if self.use_context else 0

        # Decoding RNN
        self.dec_lstm = []
        self.op = []
        # Define a series of RNN decoders according to the latent modes, decoder_size is 16
        # The input size is 80 + 32 + 64 + 8 + 2
        for k in range(self.modes):
            if not self.use_gru:
                self.dec_lstm.append(
                    torch.nn.LSTM(self.nbr_atten_embedding_size + self.dyn_embedding_size +
                                  context_feat_size + self.posi_enc_dim + self.posi_enc_ego_dim, self.decoder_size))
            else:
                self.num_layers = 2
                self.dec_lstm.append(torch.nn.GRU(
                    self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size + self.posi_enc_dim + self.posi_enc_ego_dim,
                    self.decoder_size, num_layers=self.num_layers, bidirectional=self.bi_direc))

            # assign a linear layer to the hidden state of the decoder
            self.op.append(torch.nn.Linear(self.decoder_size * self.dec_fac, 5))

            self.op[k] = self.op[k]
            self.dec_lstm[k] = self.dec_lstm[k]

        # Holds submodules in a list
        self.dec_lstm = torch.nn.ModuleList(self.dec_lstm)
        self.op = torch.nn.ModuleList(self.op)

        # Using a Linear Layer to generate possibility of modes
        self.op_modes = torch.nn.Linear(self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size,
                                        self.modes)

        # Nonlinear activations for convenience
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        # The size of input context is 640 * 48 or 320 * 96
        # We use CNN to process context into 64-dim vector
        if self.use_context:
            self.context_conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2)
            self.context_conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2)
            self.context_maxpool = torch.nn.MaxPool2d(kernel_size=(4, 2))
            self.context_conv3 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2)
            self.context_fc = torch.nn.Linear(16 * 20 * 3, context_feat_size)

    def init_rbf_state_enc(self, in_dim: int) -> None:
        """Initialize the dynamic attentional RBF encoder.
        Args:
          in_dim is the input dim of the observation, we choose 16 * 2
        """
        self.sec_in_dim = in_dim
        self.extra_pos_dim = 2

        self.sec_in_pos_dim = 2
        self.sec_key_dim = 8
        self.sec_key_hidden_dim = 32

        self.sec_hidden_dim = 32
        self.scale = 1.0
        self.slot_key_scale = 1.0
        self.num_slots = 8
        self.slot_keys = []

        # Network for computing the 'key' from output state of encoder and extra message
        # 'key' is an 8-dim vector
        self.sec_key_net = torch.nn.Sequential(
            torch.nn.Linear(self.sec_in_dim + self.extra_pos_dim, self.sec_key_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.sec_key_hidden_dim, self.sec_key_dim)
        )

        # There only exists 8 slots to be calculated
        for ss in range(self.num_slots):
            self.slot_keys.append(torch.nn.Parameter(
                self.slot_key_scale * torch.randn(self.sec_key_dim, 1, dtype=torch.float32)))
        # Holds parameters in a list, every parameter is 8 * 1
        self.slot_keys = torch.nn.ParameterList(self.slot_keys)

        # Network for encoding a scene-level contextual feature
        # Use hidden states of 8 agents to generate a nbr_atten_embedding of 80-dim
        self.sec_hist_net = torch.nn.Sequential(
            torch.nn.Linear(self.sec_in_dim * self.num_slots, self.sec_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.sec_hidden_dim, self.st_enc_hist_size)
        )

        # Encoder position of other's into a feature network, input should be normalized to ref_pos.
        # The output feature is 8-dim
        self.sec_pos_net = torch.nn.Sequential(
            torch.nn.Linear(self.sec_in_pos_dim * self.num_slots, self.sec_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.sec_hidden_dim, self.st_enc_pos_size)
        )

    def rbf_state_enc_get_attens(self, nbrs_enc: torch.Tensor, ref_pos: torch.Tensor, nbrs_pos: torch.Tensor, nbrs_info_this: List) -> List[torch.Tensor]:
        """Computing the attention over other agents.
        Args:
            nbrs_enc is hidden states of every agents
            ref_pos contains current positions of all the agents
            nbrs_info_this is a list of list of (nbr_batch_ind, nbr_id, nbr_ctx_ind)
        Returns:
            attention weights over the neighbors, which contains a few scenes.
            each scene contains num_keys * num_agents, which may means attention
        """
        assert len(nbrs_info_this) == ref_pos.shape[0]
        # Calculate 'key' for every features in nbrs_enc
        if self.extra_pos_dim > 0:
            # As for 'key', the extra message is relative position
            pos_enc = torch.zeros(nbrs_enc.shape[0], 2, device=nbrs_enc.device)
            counter = 0
            for n in range(len(nbrs_info_this)):
                for nbr in nbrs_info_this[n]:
                    pos_enc[counter, :] = nbrs_pos[nbr[0], :] - ref_pos[n, :]
                    counter += 1
            # size of Key is (nbrs_enc.shape[0], 8), every agents have corresponding keys
            Key = self.sec_key_net(torch.cat((nbrs_enc, pos_enc), dim=1))
        else:
            Key = self.sec_key_net(nbrs_enc)

        attens0 = []
        for slot in self.slot_keys:
            # Assign l2 norm to the difference between key and slots
            attens0.append(torch.exp(-self.scale * (Key - torch.t(slot)).norm(dim=1)))

        # num_keys x num_agents
        Atten = torch.stack(attens0, dim=0)
        attens = []
        counter = 0
        for n in range(len(nbrs_info_this)):
            list_of_nbrs = nbrs_info_this[n]
            counter2 = counter + len(list_of_nbrs)
            attens.append(Atten[:, counter:counter2])
            counter = counter2
        return attens

    def rbf_state_enc_hist_fwd(self, attens: List, nbrs_enc: torch.Tensor, nbrs_info_this: List) -> torch.Tensor:
        """Computes dynamic state encoding.
        Computes dynamic state encoding with precomputed attention tensor and the
        RNN based encoding.
        Args:
          attens is a list of [ [slots x num_neighbors]]
          nbrs_enc is num_agents by input_dim
        Returns:
          feature vector, size is num_scenes * (num_slots * num_encodes)
        """
        out = []
        counter = 0
        for n in range(len(nbrs_info_this)):
            list_of_nbrs = nbrs_info_this[n]
            if len(list_of_nbrs) > 0:
                counter2 = counter + len(list_of_nbrs)
                nbr_feat = nbrs_enc[counter:counter2, :]
                # matrix multiply: (num_slots * num_agents) * (num_agents * num_encode)
                out.append(torch.mm(attens[n], nbr_feat))
                counter = counter2
            else:
                out.append(torch.zeros(self.num_slots, nbrs_enc.shape[1]).to(nbrs_enc.device))
                # if no neighbors found, use all zeros.
        st_enc = torch.stack(out, dim=0).view(len(out), -1)  # num_agents by slots*enc dim
        return self.sec_hist_net(st_enc)

    def rbf_state_enc_pos_fwd(self, attens: List, ref_pos: torch.Tensor, fut_t: torch.Tensor,
                              flatten_inds: torch.Tensor, chunks: List) -> torch.Tensor:
        """Computes the features from dynamic attention for interactive rollouts.
        Args:
          attens is a list of [ [slots x num_neighbors]]
          ref_pos should be (num_agents by 2)
        Returns:
          feature vector
        """
        # convert to 'global' frame
        fut = fut_t + ref_pos
        # extract coordinate of agents in flatten_inds
        nbr_feat = torch.index_select(fut, 0, flatten_inds)
        # split agents into groups according to list chunks
        splits = torch.split(nbr_feat, chunks, dim=0)
        out = []
        for n, nbr_feat in enumerate(splits):
            # matrix multiply: (num_slots * num_agents) * (num_agents * position)
            out.append(torch.mm(attens[n], nbr_feat - ref_pos[n, :]))
        pos_enc = torch.stack(out, dim=0).view(len(attens), -1)  # num_agents by slots*enc dim
        return self.sec_pos_net(pos_enc)

    def forward_mfp(self, hist: torch.Tensor, nbrs: torch.Tensor, masks: torch.Tensor, context: Any,
                    nbrs_info: List, fut: torch.Tensor, bStepByStep: bool,
                    use_forcing: Optional[Union[None, int]] = None) -> Tuple[List[torch.Tensor], Any]:
        """Forward propagation function for the MFP
    
        Computes dynamic state encoding with precomputed attention tensor and the
        RNN based encoding.
        Args:
          hist: Trajectory history.
          nbrs: Neighbors.
          masks: Neighbors mask.
          context: contextual information in image form (if used).
          nbrs_info: information as to which other agents are neighbors.
          fut: Future Trajectory.
          bStepByStep: During rollout, interactive or independent.
          use_forcing: Teacher-forcing or classmate forcing.

        Returns:
          fut_pred: a list of predictions, one for each mode.
          modes_pred: prediction over latent modes.
        """
        use_forcing = self.use_forcing if use_forcing is None else use_forcing

        # Normalize to reference position.
        # we use last frame as anchor, and calculate relative coordinate
        # ref_pos is [num_agents, 2], hist is [seq_len, num_agents, 2]
        ref_pos = hist[-1, :, :]
        hist = hist - ref_pos.view(1, -1, 2)

        # Encode history trajectories.
        if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
            # Use GRU, ip_emb transform coordinate to 32-dim feature
            # The history trajectories is encoded into 16-dim hidden state without interaction
            # hist_enc is [2, num_agents, 16]
            _, hist_enc = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        else:
            _, (hist_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))  # hist torch.Size([16, 128, 2])

        if self.use_gru:
            # Normal function to transpose matrix
            hist_enc = hist_enc.permute(1, 0, 2).contiguous()
            # Use a Linear layer, hist_enc is transformed to [num_agents, 32]
            hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[0], -1)))
        else:
            hist_enc = self.leaky_relu(
                self.dyn_emb(hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])))  # torch.Size([128, 32])

        # Calculate number of neighbors
        # nbrs_info[0] returns a dictionary with {neighbor_index: nbs}
        num_nbrs = sum([len(nbs) for nb_id, nbs in nbrs_info[0].items()])
        if num_nbrs > 0:
            # Normalize to reference position
            # nbrs_ref_pos is [num_nbrs, 2], nbrs is [seq_len, num_nbrs, 2]
            # nbrs can be divided into num_agents parts which are the corresponding neighbors of agents
            nbrs_ref_pos = nbrs[-1, :, :]
            nbrs = nbrs - nbrs_ref_pos.view(1, -1, 2)

            # Forward pass for all neighbors.
            if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
                # nbrs_enc is [2, num_nbrs, 16]
                _, nbrs_enc = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
            else:
                _, (nbrs_enc, _) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))

            if self.use_gru:
                # nbrs_enc is [num_nbrs, 32] without dynamic embedding
                nbrs_enc = nbrs_enc.permute(1, 0, 2).contiguous()
                nbrs_enc = nbrs_enc.view(nbrs_enc.shape[0], -1)
            else:
                nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

            # Compute the attention over other agents, nbrs_enc is [num_nbrs, 32], ref_pos is [num_agents, 2]
            # nbrs_info[0] is a dictionary and the key is 0, 1, 2, ..., num_agents-1
            # which means each agent is related to a set of neighbors, the index is the value of nbr_info[0]
            # and each neighbor is also a agent in agent_enc and ref_pos
            # attens is a list with num_agents items, each item means num_key * num_cor_nbrs
            attens = self.rbf_state_enc_get_attens(nbrs_enc, ref_pos, nbrs_ref_pos, nbrs_info[0])
            # nbr_atten_enc is [num_agents, 80] which combines information of all the neighbors
            nbr_atten_enc = self.rbf_state_enc_hist_fwd(attens, nbrs_enc, nbrs_info[0])

        else:  # if have no neighbors
            attens = None
            nbr_atten_enc = torch.zeros(1, self.nbr_atten_embedding_size, dtype=torch.float32, device=masks.device)

        # context may be [num_agents, 3, 96, 320]
        # enc is [num_agents, 80 + 32 + 64]
        if self.use_context:  # context encoding
            context_enc = self.relu(self.context_conv(context))
            context_enc = self.context_maxpool(self.context_conv2(context_enc))
            context_enc = self.relu(self.context_conv3(context_enc))
            context_enc = self.context_fc(context_enc.view(context_enc.shape[0], -1))

            enc = torch.cat((nbr_atten_enc, hist_enc, context_enc), 1)
        else:
            enc = torch.cat((nbr_atten_enc, hist_enc), 1)

        ######################################################################################################
        # Use nbr_atten_enc, nbr_atten_enc and context_enc to generate probability using a Linear layer and softmax function
        modes_pred = None if self.modes == 1 else self.softmax(self.op_modes(enc))
        # call decoder function to achieve future prediction
        fut_pred = self.decode(enc, attens, nbrs_info[0], ref_pos, fut, bStepByStep, use_forcing)
        return fut_pred, modes_pred

    def decode(self, enc: torch.Tensor, attens: List, nbrs_info_this: List, ref_pos: torch.Tensor, fut: torch.Tensor,
               bStepByStep: bool, use_forcing: Any) -> List[torch.Tensor]:
        """Decode the future trajectory using RNNs.
    
        Given computed feature vector, decode the future with multimodes, using
        dynamic attention and either interactive or non-interactive rollouts.
        Args:
          enc: encoded features, one per agent.
          attens: attentional weights, list of objs, each with dimension of [8 x 4] (e.g.)
          nbrs_info_this: information on who are the neighbors
          ref_pos: the current position (reference position) of the agents.
          fut: future trajectory (only useful for teacher or classmate forcing)
          bStepByStep: interactive or non-interactive rollout
          use_forcing: 0: None. 1: Teacher-forcing. 2: classmate forcing.

        Returns:
          fut_pred: a list of predictions, one for each mode.
          modes_pred: prediction over latent modes.
        """
        if not bStepByStep:  # Non-interactive rollouts
            # out_length is 25, enc turns out to be [25, num_agent, 176]
            enc = enc.repeat(self.out_length, 1, 1)
            # pos_enc is [25, num_agents, 8 + 2], without any information from relative position
            pos_enc = torch.zeros(self.out_length, enc.shape[1], self.posi_enc_dim + self.posi_enc_ego_dim,
                                  device=enc.device)
            # enc2 is [25, num_agents, 186]
            enc2 = torch.cat((enc, pos_enc), dim=2)
            fut_preds = []
            # predict different trajectory according to modes which is achieved by different models
            for k in range(self.modes):
                # use different GRU unit, h_dec is [25, num_agents, 16], initialize hidden state and cell state as zero
                h_dec, _ = self.dec_lstm[k](enc2)
                h_dec = h_dec.permute(1, 0, 2)
                # use different linear layer, fut_pred is [num_agents, 25, 5]
                # which represents parameters of normal distribution
                fut_pred = self.op[k](h_dec)
                # fut_pred is [25, num_agents, 5]
                fut_pred = fut_pred.permute(1, 0, 2)
                # assign normalization and fut_pred remains to be [25, num_agents, 5]
                fut_pred = Gaussian2d(fut_pred)
                fut_preds.append(fut_pred)
            return fut_preds
        else:   # interactive rollouts
            batch_sz = enc.shape[0]
            inds = []
            chunks = []
            for n in range(len(nbrs_info_this)):
                # chunks reserve corresponding number of neighbors
                chunks.append(len(nbrs_info_this[n]))
                for nbr in nbrs_info_this[n]:
                    # inds reserve index of all the neighbors which can be split by chunks
                    inds.append(nbr[0])
            flat_index = torch.LongTensor(inds).to(ref_pos.device)

            fut_preds = []
            # predict multiple trajectories
            for k in range(self.modes):
                direc = 2 if self.bi_direc else 1
                # hidden is [2, num_agents, 16]
                hidden = torch.zeros(self.num_layers * direc, batch_sz, self.decoder_size).to(fut.device)
                preds: List[torch.Tensor] = []
                # predict future trajectory for each step
                for t in range(self.out_length):
                    # fut contains ground truth of future trajectory used to train LSTM units
                    if t == 0:  # Initial time step
                        if use_forcing == 0:    # no forcing
                            pred_fut_t = torch.zeros_like(fut[t, :, :])
                            ego_fut_t = pred_fut_t
                        elif use_forcing == 1:  # teacher forcing
                            pred_fut_t = fut[t, :, :]
                            ego_fut_t = pred_fut_t
                        else:   # classmate forcing
                            pred_fut_t = fut[t, :, :]
                            ego_fut_t = torch.zeros_like(pred_fut_t)
                    else:
                        # pred_fut_t is used to encode, ego_fut_t is ground truth
                        if use_forcing == 0:
                            # both neighbor and ego is from prediction
                            pred_fut_t = preds[-1][:, :, :2].squeeze()
                            ego_fut_t = pred_fut_t
                        elif use_forcing == 1:
                            # both neighbor and ego is from ground truth
                            pred_fut_t = fut[t, :, :]
                            ego_fut_t = pred_fut_t
                        else:
                            # neighbor is from ground truth, ego is from prediction
                            pred_fut_t = fut[t, :, :]
                            ego_fut_t = preds[-1][:, :, :2]

                    if attens is None:
                        pos_enc = torch.zeros(batch_sz, self.posi_enc_dim, device=enc.device)
                    else:
                        # As for teacher forcing and classmate forcing, pred_fut_t is ground truth which is relative position
                        # Otherwise, no forcing mode use last prediction
                        # pred_fut_t is [num_agents, 2]
                        # attens contains the extent of a neighbor consistent with 8 slots
                        # pos_enc is [num_agents, 8] which is relevant to the relative position of neighbors
                        pos_enc = self.rbf_state_enc_pos_fwd(attens, ref_pos, pred_fut_t, flat_index, chunks)

                    # enc_large is [1, num_agents, 176 + 8 + 2]
                    enc_large = torch.cat((enc.view(1, enc.shape[0], enc.shape[1]),
                                           pos_enc.view(1, batch_sz, self.posi_enc_dim),
                                           ego_fut_t.view(1, batch_sz, self.posi_enc_ego_dim)), dim=2)

                    # calculate one frame for one time, and reserve output and hidden state
                    out, hidden = self.dec_lstm[k](enc_large, hidden)
                    pred = Gaussian2d(self.op[k](out))
                    # pred is [1, num_agents, 5], and in next frame we use the position with highest probability
                    preds.append(pred)
                fut_pred_k = torch.cat(preds, dim=0)
                fut_preds.append(fut_pred_k)
            return fut_preds
