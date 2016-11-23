require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model.RNN'
require 'model.decoder'
require 'util.misc' -- share_params
require 'util.encoder'
require 'optim'

local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'

-- opt
opt = {}
opt.data_dir = 'data/tinyshakespeare'
opt.batch_size =42
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
opt.model = 'birnn'

-- 1: train 2:val 3: text
-- loader:next_batch(num)

-- master cell for all other cell to share params with
master_cell = RNN.create(opt)
master_decoder = decoder:create(opt)
params, gradParams = master_cell:getParameters()

-- create a rnn chain
model = {}
model_b = {}
decoders = {}
criterion = {}
for i = 1,opt.seq_length do
  local cell = RNN.create(opt)
  local cell_b = RNN.create(opt)
  local crit = nn.ClassNLLCriterion()
  local d = decoder:create(opt)
  share_params(cell,master_cell)
  share_params(cell_b, master_cell)
  share_params(d, master_decoder)
  table.insert(model, cell)
  table.insert(model_b, cell_b)
  table.insert(decoders, d)
  table.insert(criterion,crit)
end

final_lost = 0
-- feval
function feval(p)
-- This part is important, dont skip it
  if p ~= params then
	  params:copy(p)
  end
  gradParams:zero()
-- important for feval

  -- x data
  -- y label
  x, y = loader:next_batch(1)
  x, y = prepro(x,y)

  -- declare something here
  local loss = 0
  local h = {}
  local h_b = {}
  h_0 = torch.Tensor(opt.batch_size,opt.rnn_size):zero()
  h[0] = h_0
  h_b[opt.seq_length+1] = h_0

  -- forward pass
  for t = 1, opt.seq_length do
    model[t]:training()
    local input_x = encoder.oneHot(x[t],vocab_size) -- convert into one hot vector
    local output = model[t]:forward({input_x, h[t-1]}) --x[t]:unfold(1,1,1) is element t in sequence of each batch
    h[t] = output
  end

  -- forward pass of reserve direction
  for t = opt.seq_length, 1, -1 do
    model_b[t]:training()
    local input_x = encoder.oneHot(x[t],vocab_size) -- convert into one hot vector
    local output = model_b[t]:forward({input_x, h_b[t+1]}) --x[t]:unfold(1,1,1) is element t in sequence of each batch
    h_b[t] = output
  end

  -- h[t]:size() 42 128

  -- calculate predict
  local predict = {}
  local obj_grad = {}
  local rep_grad = {}
  for t = 1, opt.seq_length do
    predict[t] = decoders[t]:forward({h[t],h_b[t]})
    loss = loss + criterion[t]:forward(predict[t],y[t])
  end

  -- predict size: 42 65


  -- backward on criterion and decoders
  for t = 1, opt.seq_length do
    obj_grad[t] = criterion[t]:backward(predict[t],y[t])
    rep_grad[t] = decoders[t]:backward({h[t],h_b[t]}, obj_grad[t])
  end

  -- backward on model
  for t = opt.seq_length, 1, -1 do
    local input_x = encoder.oneHot(x[t],vocab_size)
    print(input_x:cdata())
    print(h[t-1]:cdata())
    print(rep_grad[t][1]:cdata())
    model[t]:backward({input_x, h[t-1]},{rep_grad[t][1]}) -- TODO: error here
  end



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
