require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model.SimpleRNN'
require 'util.misc' -- share_params
require 'util.encoder'
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

opt.input_size = vocab_size
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

final_lost = 0
-- feval
function feval(x)
-- This part is important, dont skip it
  if x ~= params then
	  params:copy(x)
  end
  gradParams:zero()
-- important for feval

  local predict ={} -- y^
  local h = {} -- h
  local loss = 0

    --
    -- x data
    -- y label
    x, y = loader:next_batch(1)
    x, y = prepro(x,y)
    h_0 = torch.Tensor(opt.batch_size,opt.rnn_size):zero()
    h[0] = h_0

    -- forward pass
    for t = 1, opt.seq_length do
      model[t]:training()
      local input_x = encoder.oneHot(x[t],vocab_size) -- convert into one hot vector
      local output = model[t]:forward({input_x, h[t-1]}) --x[t]:unfold(1,1,1) is element t in sequence of each batch
      h[t] = output[1]
      predict[t] = output[2]
      loss = loss + criterion[t]:forward(predict[t],y[t])
    end

    -- backward pass
    local dh = {}
    dh[opt.seq_length] = h_0
    for t = opt.seq_length, 1, -1 do
      local doutput = criterion[t]:backward(predict[t], y[t])
      local input_x = encoder.oneHot(x[t],vocab_size) -- convert into one hot vector
      local d = model[t]:backward({input_x,h[t-1]},{dh[t],doutput})
      dh[t-1] = d[2]
    end
    loss = loss / opt.seq_length
--    print(loss)
    final_lost = loss
    return loss, gradParams
end



optimState ={
  learningRate = 0.01
}

local timer = torch.Timer()
for epoch = 1, 1000 do
  optim.sgd(feval,params, optimState)
end
print(timer:time().real .. ' seconds')
print('lost '..final_lost)

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

-- create ivocab
ivocab = {}
ivocab = create_ivocab(vocab)

function evaluate(seedtext)
    master_cell:evaluate()
    local h = torch.Tensor(opt.rnn_size):zero()
    output_text = ""
    for c in seedtext:gmatch'.' do
      prev_char = torch.Tensor{vocab[c]}
      local input_x = encoder.oneHot(prev_char, vocab_size)
      outputs = master_cell:forward({input_x,h});
      h = outputs[1]
      pred = outputs[2]
      val, idx = torch.max(pred,1)
      output_text = output_text..ivocab[idx[1]]
    end
    print(output_text)

end

evaluate("We are accounted poor citizens, the patricians good.What authority surfeits on would relieve us: if theywould yield us but the superfluity, while it werewholesome, we might guess they relieved us humanely;but they think we are too dear: the leanness thatafflicts us, the object of our misery, is as aninventory to particularise their abundance; oursufferance is a gain to them Let us revenge this withour pikes, ere we become rakes: for the gods know Ispeak this in hunger for bread, not in thirst for revenge.")
