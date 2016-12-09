--[[ here's a kmeans algo on one gather, with clustering ]]--

require 'unsup'

ns = 1501
ntr = 45

timer = torch.Timer()

file = torch.DiskFile('../data_loading/test_dat/four_gathers.rsf@', 'r')
file:binary()
raw = file:readFloat(ns*ntr)
file:close()

dat = torch.Tensor(ns,ntr)

for j = 1, ntr do
        for i = 1, ns do
                dat[i][j] = raw[i + (j-1)*ns]
        end
end

-- condition data with zero mean and unit variance
dat:add(-dat:mean())
dat:div(dat:std())

-- k means
k = 5		-- # of centroids
iter = 200	-- # of iteratiosn
bsz = 10	-- batchsize
centroids, totalcounts = unsup.kmeans(dat, k, iter, bsz, nil, true)

-- clustering
clst = torch.Tensor(ns)
dist = torch.Tensor(k)

-- calc Euclidean distance to centroids
for i = 1, ns do
	for j = 1, k do
		dist[j] = torch.sqrt((centroids[j] - dat[i]):pow(2):sum())
	end
	
	-- cluster by minimum distance
	dist, idx = dist:sort()
	clst[i] = idx[1]
end

--print(clst)
print(timer:time().real)
