--[[ 'bbimgath' the data has 1501 time samples and 45 traces per gather]]--

require 'unsup'

ns = 1501
ng = 45
ntr = 7101405

-- designate I/O
infile = torch.DiskFile('/home/ubuntu/bbimgath/gathers.rsf@', 'r')
infile:binary()

os.remove('/home/ubuntu/bbimgath/eigenvectors')
outfile = torch.DiskFile('/home/ubuntu/bbimgath/eigenvectors', 'w')
outfile:binary()

-- initialize
dat = torch.Tensor(ns,ng)
counter = 10

for k = 1, ntr, ng do
	-- progress
	if k/ntr > counter then
		print('percentage complete: ', k/ntr, '%\n')
		counter = counter + 10
	end
	
	-- load one gather into mem
	infile:seek(k)
	raw = infile:readFloat(ns*ng)

	-- build 2D tensor
	for j = 1, ng do
        	for i = 1, ns do
                	dat[i][j] = raw[i + (j-1)*ns]
	        end
	end

	-- perform covariance PCA
	e, v = unsup.pcacov(dat)

	-- write out eigenvectors to file
	for j = 1, ng do
		for i = 1, ng do
			outfile:writeFloat(v[i][j])
		end
	end

end

-- clean
infile:close()
outfile:close()

