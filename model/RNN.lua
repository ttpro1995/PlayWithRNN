require 'torch'
require 'nn'
require 'nngraph'

RNN = {}
-- rnn without decoder
function RNN.create(opt)
  local x = nn.Identity()()
  local h_prev = nn.Identity()()

  local Wh = nn.Linear(opt.rnn_size,opt.rnn_size)({h_prev})
  local Ux = nn.Linear(opt.input_size, opt.rnn_size)({x})

  local a = nn.CAddTable()({Wh, Ux})
  local h = nn.Tanh()({a})

  return nn.gModule({x,h_prev},{h})
end
