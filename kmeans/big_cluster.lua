--[[ 'bbimgath' the data has 1501 time samples and varying traces per gather.
	use qlua <script> instead of th <script> to run this bc of `image.display`]]--

require 'unsup'
require 'image'

ns = 1501
ntr = 45080
ng = 46

-- designate I/O
infile = torch.DiskFile('/home/ubuntu/bbimgath/wbb_gathers@data@', 'r')
infile:binary()

outfile = torch.DiskFile('/home/ubuntu/bbimgath/clustered', 'w')
outfile:binary()

-- initialize
dat = torch.Tensor(ns,ng)
timer = torch.Timer()
clst = torch.Tensor(ns)
k = 5		-- # of centroids
iter = 200	-- # of iterations
bsz = 10	-- batchsize
dist = torch.Tensor(k)
counter = 1


-- k means centroid calculation
	-- intitialize
treval = 2300
raw = infile:readFloat(ns * treval)
local big = torch.Tensor(ns*treval/ng, ng)

for k = 1, treval, ng do
	for j = 1, ng do
		for i = 1, ns do
			big[i + ((k-1)*ns/ng)][j] = raw[i + (j-1)*ns + (k-1)*ns]
		end
	end
end

	-- condition data with zero mean and unit variance
big:add(-big:mean())
big:div(big:std())

	-- k means
centroids, totalcounts = unsup.kmeans(big, k, iter, bsz, nil, true)

	-- clean
big = nil
collectgarbage()

-- clustering gather-per-gather
for kk = 1, ntr, ng do
	-- debug
	--print(counter)
	--print(kk)
	counter = counter + 1

	-- load one gather into mem
	if kk == 1 then
		infile:seek(1)
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

	-- clustering
		-- calc Euclidean distance to centroids
	for i = 1, ns do
		for j = 1, k do
			dist[j] = torch.sqrt((centroids[j] - dat[i]):pow(2):sum())
		end

		-- cluster by minimum distance
		dist, idx = dist:sort()
		clst[i] = idx[1]
	end

	-- write out clustered data
	for i = 1, ns do
		outfile:writeFloat(clst[i])
	end
end

print('timing = ',timer:time().real)

--image.display(clst)
--print(clst)

-- clean
infile:close()
outfile:close()
