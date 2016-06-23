-- 라이브러리
require 'loadcaffe'
require 'xlua'
require 'optim'
mnist = require 'mnist'

-- to train lenet network please follow the steps
-- provided in CAFFE_DIR/examples/mnist
-- 학습을 위한 절차
-- 프로토텍스트 와 바이너리 데이타 경로 지정
prototxt = '/opt/caffe/examples/mnist/lenet.prototxt'
binary = '/opt/caffe/examples/mnist/lenet_iter_10000.caffemodel'

-- this will load the network and print it's structure
-- 네트웍 로딩 및 구조 인쇄
net = loadcaffe.load(prototxt, binary)

-- load test data
-- 학습용 데이타 로딩.
testData = mnist.testdataset()

-- preprocess by dividing by 256
-- 0~1 사이로 정규화
images = testData.data:float():div(256)

-- 아규먼트로 쿠다 모드
if arg[1] == 'cuda' then
  net:cuda()
  images = images:cuda()
else
  net:float()
end

-- will be used to print the results
-- 결과 프린트하기, 컨퓨전 매트릭스
confusion = optim.ConfusionMatrix(10)

-- ROC Curve 를 추가하기.
for i=1,images:size(1) do
  _,y = net:forward(images[i]:view(1,28,28)):max(1)
  confusion:add(y[1], testData.label[i]+1)
end

-- that's all! will print the error and confusion matrix
-- 출력하기
print(confusion)
