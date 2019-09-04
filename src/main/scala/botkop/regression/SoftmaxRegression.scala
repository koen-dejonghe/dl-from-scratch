package botkop.regression

import botkop.{numsca => ns}
import botkop.data.FashionMnistDataLoader
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

object SoftmaxRegression extends App with LazyLogging {

  val batchSize = 256

  logger.info("reading training data set")
  val trainDl = new FashionMnistDataLoader("train", batchSize)
  logger.info("training data set read")

  logger.info("reading test data set")
  val testDl = new FashionMnistDataLoader("test", batchSize)
  testDl.normalize(trainDl.meanImage)
  logger.info("test data set read")

  val numFeatures = trainDl.numFeatures
  val numClasses = 10

  val w = Variable(ns.randn(numFeatures, numClasses) * 0.01)
  val b = Variable(ns.zeros(numClasses))

  def net(x: Variable): Variable = (x dot w) + b

  def loss(yHat: Variable, y: Variable): Variable = scorch.softmaxLoss(yHat, y)

  def sgd(params: Seq[Variable], lr: Double, batchSize: Int): Unit =
    params.foreach { p =>
      p.data -= lr * p.grad.data / batchSize
      p.grad.data := 0.0
    }

  def evaluate(dl: DataLoader, net: Variable => Variable): (Double, Double) = {
    val (l, a) = dl.foldLeft(0.0, 0.0) {
      case ((lossAcc, accuracyAcc), (x, y)) =>
        val output = net(x)
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(guessed == y.data)
        val cost = loss(output, y).data.squeeze()
        (lossAcc + cost, accuracyAcc + accuracy)
    }
    (l / dl.numBatches, a / dl.numSamples)
  }

  val numEpochs = 15
  val lr = 0.3

  (0 until numEpochs) foreach { epoch =>
    logger.info(s"starting epoch $epoch")
    trainDl.foreach {
      case (x, y) =>
        val yh = net(x)
        val l = loss(yh, y)
        l.backward()
        sgd(Seq(w, b), lr, batchSize) // update parameters using their gradient
    }
    val (l, a) = evaluate(testDl, net)
    logger.info(s"epoch $epoch\tloss: $l\taccuracy: $a")
  }

}
