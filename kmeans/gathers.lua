--[[ 'bbimgath' the data has 1501 time samples and varying traces per gather.
	use qlua <script> instead of th <script> to run this bc of `image.display`]]--

require 'unsup'
require 'image'

ns = 1501
ntr = 184
ng = 46

-- designate I/O
infile = torch.DiskFile('/home/gram/ava-class/data_loading/test_dat/four_gathers.rsf@', 'r')
infile:binary()

-- initialize
dat = torch.Tensor(ns,ng)
timer = torch.Timer()
clst = torch.Tensor(ns, ntr/ng)
k = 5		-- # of centroids
iter = 100	-- # of iteratiosn
bsz = 10	-- batchsize
dist = torch.Tensor(k)
counter = 1

for kk = 1, ntr, ng do
	-- load one gather into mem
	if kk == 1 then
		raw = infile:readFloat(ns*ng)
	else
		infile:seek(kk*ns*4 + 1)
		raw = infile:readFloat(ns*ng)
	end

	-- build 2D tensor
	for j = 1, ng do
        	for i = 1, ns do
                	dat[i][j] = raw[i + (j-1)*ns]
	        end
	end

	-- condition data with zero mean and unit variance
	dat:add(-dat:mean())
	dat:div(dat:std())

	-- k means
	centroids, totalcounts = unsup.kmeans(dat, k, iter, bsz, nil, true)

	-- clustering
		-- calc Euclidean distance to centroids
	for i = 1, ns do
		for j = 1, k do
			dist[j] = torch.sqrt((centroids[j] - dat[i]):pow(2):sum())
		end

		-- cluster by minimum distance
		dist, idx = dist:sort()
		clst[i][counter] = idx[1]
	end
	counter = counter + 1
end

print('timing = ',timer:time().real)

image.display(clst)
--print(clst)

-- clean
infile:close()
