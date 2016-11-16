require 'torch'

encoder = {}


--[[
input: an 1D array.
Each element in input is a number -> vocab -> word.
output: a matrix.
each element in input -> one hot row in output matrix
]]--
function encoder.oneHot(input, dimension)
  local n_sample = input:size(1)
  local output = torch.Tensor(input:size(1),dimension):zero()
  for i = 1, n_sample do
    output[i][input[i]] = 1
  end

  return output
end
