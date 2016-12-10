--[[ `image.display` is dependent on qtlua, which is bugged. To run this program
	you've got to use qlua <script>, not th <script> ]]--	

require 'image'
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

image.display(dat)


dat:add(-dat:mean())
dat:div(dat:std())

vals, vects = unsup.pcacov(dat)

image.display(vects)

cmprsd = vects[{{},{-2,-1}}]

new = torch.mm(torch.mm(dat,cmprsd), cmprsd:t())

image.display(new)

