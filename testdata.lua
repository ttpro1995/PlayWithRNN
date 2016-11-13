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
opt.batch_size =10
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

opt.input_size = 1
opt.output_size = vocab_size


x, y = loader:next_batch(1)
x, y = prepro(x,y)

a = x:narrow(2,1,1)
b = y:narrow(2,1,1):squeeze()
