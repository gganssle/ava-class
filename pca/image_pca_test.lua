-- I'm using this to test the order of the eigenvectors coming out of unsup.pca

--[[ `image.display` is dependent on qtlua, which is bugged. To run this program
	you've got to use qlua <script>, not th <script> ]]--	

require 'image'
require 'unsup'

old = image.lena()
--image.display(old)

gray = old[3]

gray:add(-gray:mean())
gray:div(gray:std())

vals, vects = unsup.pcacov(gray)

--image.display(vects)

cmprsd = vects[{{-500,-1},{}}]
cmprsd = vects[{{},{-10,-1}}]

new = torch.mm(torch.mm(gray,cmprsd), cmprsd:t())

image.display(gray)
image.display(new)

