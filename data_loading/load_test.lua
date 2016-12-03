ns = 1501
ntr = 187

file = torch.DiskFile('test_dat/four_gathers@data@', 'r')
file:binary()
raw = file:readFloat(ns*ntr)
file:close()

dat = torch.Tensor(ns,ntr)

for j = 1, ntr do
	for i = 1, ns do
		dat[i][j] = raw[i + (j-1)*ns]
	end
end

print(dat[{{},{187}}])
