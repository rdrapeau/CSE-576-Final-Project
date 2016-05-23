require 'paths'
require 'cjson'
require 'torch'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Pose extraction using stacked hourglass network.')
cmd:text()
cmd:text('Options:')

cmd:option('-inDir', '.', 'Input directory')
cmd:option('-model', './umich-stacked-hourglass.t7', 'Path to model.')
cmd:option('-imgDim', 256, 'Image dimension for input to NN')
cmd:option('-gpuSelect', 1, 'The GPU index. If you have 4 GPUs, 1-4 are available')
cmd:text()

opt = cmd:parse(arg or {})
model = torch.load(opt.model)

cutorch.setDevice(opt.gpuSelect)
model = model:cuda()

function getPreds(hms)
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(
    	hms:size(1),
    	hms:size(2),
    	hms:size(3) * hms:size(4)
    ), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x)
    	return (x - 1) % hms:size(4) + 1
    end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(.5)

    return preds
end

function loadImage(impath, sX, sY)
	local im = image.load(impath)

	-- Resize
	im = image.scale(im, sX, sY)

	return im
end

function listDirectory(path, ext)
	local filePaths = {};
	for f in paths.files(path, ext) do
		table.insert(filePaths, f)
	end

	return filePaths
end

local imagePaths = listDirectory(opt.inDir, '.jpg')
local result = {}

print('Estimating ' .. #imagePaths .. ' poses')

for id, filePath in ipairs(imagePaths) do
	local img = loadImage(opt.inDir .. '/' .. filePath, opt.imgDim, opt.imgDim)
    local out = model:forward(img:view(1, 3, opt.imgDim, opt.imgDim):cuda())
    cutorch.synchronize()
    local hm = out[2][1]:float()
    hm[hm:lt(0)] = 0

    local preds_hm = getPreds(hm)
    result[filePath] = torch.Tensor(32)
    result[filePath]:copy(preds_hm:view(32))

	print('Done with ' .. filePath)
    print(result[filePath])
end

print(result)