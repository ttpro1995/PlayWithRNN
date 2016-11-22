require 'torch'
require 'nn'
require '../model/decoder'


function test_decoder()
  opt = {}
  opt.rnn_size = 10
  opt.input_size = 5
  opt.output_size = 5

  -- rnn
  opt.model = 'rnn'
  rnn_decoder = decoder:create(opt)
  hidden_rnn = torch.randn(10)
  output1 = rnn_decoder:forward(hidden_rnn)

  -- birnn
  opt.model = 'birnn'
  birnn_decoder = decoder:create(opt)
  h1 = torch.randn(10)
  h2 = torch.randn(10)
  output2 = birnn_decoder:forward({h1,h2})
end

test_decoder()
