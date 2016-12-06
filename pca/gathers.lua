--[[ 'bbimgath' the data has 1501 time samples and varying traces per gather]]--

require 'unsup'

ns = 1501
--DEV: FIX GATHER TERMINATOR---------------------------------------
ng = 45
ntr = 7101405

-- designate I/O
infile = torch.DiskFile('/home/ubuntu/bbimgath/gathers.rsf@', 'r')
infile:binary()

os.remove('/home/ubuntu/bbimgath/eigenvectors')
outfile = torch.DiskFile('/home/ubuntu/bbimgath/eigenvectors', 'w')
outfile:binary()

os.remove('/home/ubuntu/bbimgath/parfile')

-- initialize
dat = torch.Tensor(ns,ng)
counter = 1
timer = torch.Timer()

for k = 1, ntr, ng do
	-- progress
	if k/ntr > counter then
		parfile = io.open('/home/ubuntu/bbimgath/parfile', 'a')
		print('percentage complete: ', k/ntr, '%\n')
		parfile:write('percentage complete: ', k/ntr, '%\n')
		counter = counter + 1
		parfile:close()
	end
	
	-- load one gather into mem
	if k == 1 then
		raw = infile:readFloat(ns*ng)
	else
		infile:seek(k*ns*4 + 1)
		raw = infile:readFloat(ns*ng)
	end

	-- build 2D tensor
	for j = 1, ng do
        	for i = 1, ns do
                	dat[i][j] = raw[i + (j-1)*ns]
	        end
	end	

	-- condition data
--[[	dat:add(-dat:mean())
	dat:div(dat:std()) ]]--

	-- perform covariance PCA
	e, v = unsup.pcacov(dat)

	-- write out eigenvectors to file
	for j = 1, ng do
		for i = 1, ng do
			outfile:writeFloat(v[i][j])
		end
	end

end

-- timing
print('total elapsed time = ', timer:time().real)
parfile = io.open('/home/ubuntu/bbimgath/parfile', 'a')
parfile:write('total elapsed time = ', timer:time().real)

-- clean
infile:close()
outfile:close()
parfile:close()
