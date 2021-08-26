import torch
from torch import nn


def j_transition(x, h, U, W, b):
    j_uotput = torch.tanh(torch.einsum("abc,bcd->abd", h, U) +\
                          torch.einsum("abc,cbd->acd", x, W) + b)
    return j_uotput

def gate_transition(x,fc_gate):
    gate_otput = torch.sigmoid(fc_gate(x))
    return gate_otput

def exp_method(x):
    x = torch.exp(x)
    x = x / torch.sum(x, dim=1, keepdim=True)
    return x

class ILSTM_SV(nn.Module):
    def __init__(self, input_dim,timestep,output_dim, hidden_size, std=0.01):
        super().__init__()
        ######weight parameter for j
        self.W_j = nn.Parameter(torch.randn(input_dim, 1, hidden_size) * std)
        self.U_j = nn.Parameter(torch.randn(input_dim, hidden_size, hidden_size) * std)
        self.b_j = nn.Parameter(torch.randn(input_dim, hidden_size) * std)
        ######weight and bias parameter for gate
        self.W_i = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        self.W_f = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        self.W_o = nn.Linear(input_dim * (hidden_size + 1), input_dim * hidden_size)
        ##########################################
        self.FC_mulit_FV = nn.Linear(hidden_size, output_dim)
        self.FC_aten_predicor = nn.Linear(2 * hidden_size, 1)
        self.FC_predicor_output = nn.Linear(timestep, 1)
        self.FC_predicor_vect = nn.Linear(2 * hidden_size, 1)
        self.FC_temporal_output = nn.Linear(input_dim, 1)
        self.FC_temporal_vect = nn.Linear(input_dim+hidden_size, 1)
        self.FC_aten_temporal = nn.Linear(input_dim+hidden_size, 1)
        self.FC_final_output=nn.Linear(2 * output_dim, 1)

        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        outputs = []
        h_t = torch.zeros(x.shape[0], self.input_dim, self.hidden_size).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.hidden_size).cuda()
        #######LSTM#########
        for t in range(x.shape[1]):
            x_timstep=x[:,t,:].unsqueeze(1)
            gate_input= torch.cat([x[:, t, :], h_t.view(h_t.shape[0], -1)], dim=1)
            ####(16,5,32) (5,32,32) +(16,1,5) (5,1,32)+ (5,32)    (16,5,32)
            j_t = j_transition(x_timstep,h_t,self.U_j,self.W_j,self.b_j)
            i_t = gate_transition(gate_input,self.W_i)
            f_t = gate_transition(gate_input,self.W_f)
            o_t = gate_transition(gate_input,self.W_o)
            c_t = c_t * f_t + i_t * j_t.reshape(j_t.shape[0], -1)
            h_t = (o_t * torch.tanh(c_t)).view(h_t.shape[0], self.input_dim, self.hidden_size)
            outputs += [h_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        outputs = self.dropout(outputs)
        #######attention-muil feature vectors-LSTM#########
        mulit_FV_aten=self.FC_mulit_FV(outputs)
        mulit_FV_aten = exp_method(mulit_FV_aten)
        mulit_FV_aten_input=mulit_FV_aten*outputs
        #################################  predicor attention weight "comb_vect"
        predicor_aten_input = mulit_FV_aten_input.permute(0, 2, 3, 1)
        predicor_aten_output = self.FC_predicor_output(predicor_aten_input)
        predicor_comb_vect = torch.cat([predicor_aten_output.squeeze(), h_t], dim=2)
        predicor_comb_vect_FCoutput = self.FC_predicor_vect(predicor_comb_vect)
        predicor_aten = self.FC_aten_predicor(predicor_comb_vect)
        predicor_aten = exp_method(predicor_aten)
        predicor_prediction = torch.sum(predicor_aten * predicor_comb_vect_FCoutput, dim=1)
        #################################  temporal attention weight "comb_vect"
        temporal_aten_input = mulit_FV_aten_input.permute(0, 1, 3, 2)
        temporal_aten_output = self.FC_temporal_output(temporal_aten_input)
        temporal_comb_vect = torch.cat([temporal_aten_output.squeeze(), x], dim=2)
        temporal_comb_vect_FCoutput = self.FC_temporal_vect(temporal_comb_vect)
        temporal_aten = self.FC_aten_temporal(temporal_comb_vect)
        temporal_aten = exp_method(temporal_aten)
        temporal_prediction = torch.sum(temporal_aten * temporal_comb_vect_FCoutput, dim=1)
        ################################# final prediction.
        final_input = torch.cat([predicor_prediction, temporal_prediction], dim=1)
        prediction=self.FC_final_output(final_input)

        return prediction, mulit_FV_aten, predicor_aten,temporal_aten
