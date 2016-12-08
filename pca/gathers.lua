--[[ 'bbimgath' the data has 1501 time samples and varying traces per gather]]--

require 'unsup'

ns = 1501
--DEV: FIX GATHER TERMINATOR---------------------------------------
ng = 45
ntr = 7101405

--ntr = 187
--ntr = 180
--ng = 45

-- designate I/O
infile = torch.DiskFile('/home/ubuntu/bbimgath/gathers.rsf@', 'r')
--infile = torch.DiskFile('/home/ubuntu/ava-class/data_loading/test_dat/four_gathers.rsf@', 'r')
infile:binary()

os.remove('/home/ubuntu/bbimgath/eigenvectors')
outfile = torch.DiskFile('/home/ubuntu/bbimgath/eigenvectors', 'w')
outfile:binary()

os.remove('/home/ubuntu/bbimgath/parfile')
parfile = io.open('/home/ubuntu/bbimgath/parfile', 'w')
parfile:write('Start time (in GMT): ', os.date(), '\n')
parfile:flush()

-- initialize
dat = torch.Tensor(ns,ng)
counter = 0.01
timer = torch.Timer()

for k = 1, ntr, ng do
	-- torch.DiskFile:flush() workaround [1/2]
	outfile = torch.DiskFile('/home/ubuntu/bbimgath/eigenvectors', 'w')
	outfile:binary()
	outfile:seekEnd()

	-- progress
	if k/ntr > counter then
		print('percentage complete: ', k/ntr, '%\n')
		parfile:write('time = ', os.date(), ' ')
		parfile:write('percentage complete: ', k/ntr, '%\n')
		counter = counter + 0.01
		parfile:flush()
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

	-- condition data with zero mean and unit variance
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
	
	-- torch.DiskFile:flush() workaround [2/2]
	outfile:close()

end

-- timing
print('total elapsed time = ', timer:time().real)
parfile:write('total elapsed time = ', timer:time().real)

-- clean
infile:close()
--outfile:close()
parfile:close()
