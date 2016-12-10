--[[ This is a PCA test on a single gather using the bbimgath data
	the data has 1501 time samples ]]--

--[[ NOTE 20161205: gathers have varying number. I need to figure out the designator ]]--

require 'unsup'

ns = 1501
ntr = 45

file = torch.DiskFile('/home/gram/ava-class/data_loading/test_dat/four_gathers.rsf@', 'r')
file:binary()
raw = file:readFloat(ns*ntr)
file:close()

dat = torch.Tensor(ns,ntr)

for j = 1, ntr do
        for i = 1, ns do
                dat[i][j] = raw[i + (j-1)*ns]
        end
end


e, v = unsup.pcacov(dat)

--print('size e = ', #e, '\n')
--print('size v = ', v:size(1), 'x', v:size(2), '\n')

--print(v)
print(e)


