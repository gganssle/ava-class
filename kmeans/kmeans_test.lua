--[[ this is a test of unsup.kmeans. I'm using the test data here as to
	not consume a ton of memory ]]--

require 'unsup'

ns = 1501
ntr = 187

file = torch.DiskFile('../data_loading/test_dat/four_gathers@', 'r')
file:binary()
raw = file:readFloat(ns*ntr)
file:close()

dat = torch.Tensor(ns,ntr)

for j = 1, ntr do
        for i = 1, ns do
                dat[i][j] = raw[i + (j-1)*ns]
        end
end


