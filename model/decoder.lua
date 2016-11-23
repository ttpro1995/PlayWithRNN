require 'torch'
require 'nn'
require 'nngraph'

decoder = {}
--[[
opt.rnn_size
opt.output_size
opt.input_size
opt.model
]]--
function decoder:create(opt)

  -- o = Vh + c
  -- y = softmax(o)

  if (opt.model == 'rnn') then
    local h = nn.Identity()()
    local o = nn.Linear(opt.rnn_size,opt.output_size)({h})
    local y = nn.LogSoftMax()({o})
    return nn.gModule({h},{y})
  end

  if (opt.model == 'birnn') then
    local h1 = nn.Identity()()
    local h2 = nn.Identity()()
    local h = nn.JoinTable(2)({h1,h2}) -- join by column (dimension 2)  
    local o = nn.Linear(opt.rnn_size*2 , opt.output_size)({h})
    local y = nn.LogSoftMax()({o})
    return nn.gModule({h1, h2},{y})
  end

end
