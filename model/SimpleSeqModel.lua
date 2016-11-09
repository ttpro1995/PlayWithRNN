require 'nn'
require 'torch'

function create_seq(opt)
  local model = nn.Sequential()
  model:add(nn.Linear(opt.size_in,opt.size_out))
  model:add(nn.LogSoftMax())
  return model
end
