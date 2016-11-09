require 'torch'
require 'nn'
require 'nngraph'

RNN = {}

function RNN.create(opt)
  local x = nn.Identity()()
  local h_prev = nn.Identity()()

  local Wh = nn.Linear(opt.rnn_size,opt.rnn_size)({h_prev})
  local Ux = nn.Linear(opt.input_size, opt.rnn_size)({x})

  local a = nn.CAddTable()({Wh, Ux})
  local h = nn.Tanh()({a})

  -- the decoder
  local o = nn.Linear(opt.rnn_size,opt.input_size)({h})
  local y = nn.LogSoftMax()({o})


  return nn.gModule({x,h_prev},{h,y})
end
