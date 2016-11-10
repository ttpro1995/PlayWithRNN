require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model.SimpleRNN'
require 'util.misc' -- share_params
require 'optim'

local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'

-- opt
opt = {}
opt.data_dir = 'data/tinyshakespeare'
opt.batch_size =50
opt.seq_length =50
opt.train_frac =0.95
opt.val_frac =0.05

opt.rnn_size = 128


 test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
 split_sizes = {opt.train_frac, opt.val_frac, test_frac}


-- load data
 loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
 vocab_size = loader.vocab_size  -- the number of distinct characters
 vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)

opt.input_size = 50
opt.output_size = vocab_size

-- 1: train 2:val 3: text
-- loader:next_batch(num)

-- master cell for all other cell to share params with
master_cell = RNN.create(opt)
params, gradParams = master_cell:getParameters()

-- create a rnn chain
model = {}
criterion = {}
for i = 1,opt.seq_length do
  local cell = RNN.create(opt)
  local crit = nn.ClassNLLCriterion()
  share_params(cell,master_cell)
  table.insert(model, cell)
  table.insert(criterion,crit)
end

-- feval
function feval(params)
  local predict ={} -- y^
  local h = {} -- h
  local loss = 0

    --
    -- x data
    -- y label
    x, y = loader:next_batch(1)
    x, y = prepro(x,y)
    h_0 = torch.Tensor(opt.rnn_size):zero()
    h[0] = h_0

    -- forward pass
    for t = 1, opt.seq_length do
      local output = model[t]:forward({x[t], h[t-1]})
      h[t] = output[1]
      predict[t] = output[2]
      loss = loss + criterion[t]:forward(predict[t],y[t])
    end

    -- backward pass
    local dh = {}
    dh[opt.seq_length] = h_0
    for t = opt.seq_length, 1, -1 do
      local doutput = criterion[t]:backward(predict[t], y[t])
      local d = model[t]:backward({x[t],h[t-1]},{dh[t],doutput})
      dh[t-1] = d[2]
    end
    loss = loss / opt.seq_length

    return loss, gradParams
end



optimState ={
  learningRate = 0.01
}

for epoch = 1, 100 do
  optim.sgd(feval,params, optimState)
end


function test()
  predict ={} -- y^
  h = {} -- h
  local loss = 0

    --
    -- x data
    -- y label
    x, y = loader:next_batch(1)
    x, y = prepro(x,y)
    h_0 = torch.Tensor(opt.rnn_size):zero()
    h[0] = h_0

    --forward pass
    for t = 1, opt.seq_length do
      local output = model[t]:forward({x[t], h[t-1]}) 
      h[t] = output[1]
      predict[t] = output[2]
      loss = loss + criterion[t]:forward(predict[t],y[t])
    end

    -- backward pass
    local dh = {}
    dh[opt.seq_length] = h_0
    for t = opt.seq_length, 1, -1 do
      local doutput = criterion[t]:backward(predict[t], y[t])
      local d = model[t]:backward({x[t],h[t-1]},{dh[t],doutput})
      dh[t-1] = d[2]
    end
    loss = loss / opt.seq_length

    return loss, gradParams
end

test()
--
