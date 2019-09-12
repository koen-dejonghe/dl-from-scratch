package botkop.regression

import java.util.Locale

import botkop.autograd.{SoftmaxLoss, Variable}
import botkop.{numsca => ns}
import botkop.data.{DataLoader, FashionMnistDataLoader}
import com.typesafe.scalalogging.LazyLogging

object SoftmaxRegression extends App with LazyLogging {

  Locale.setDefault(Locale.US)
  val batchSize = 256
  // val take = Some(100)
  val take = None

  logger.info("reading training data set")
  val trainDl = new FashionMnistDataLoader("train", batchSize, take=take)

  logger.info("reading test data set")
  val testDl = new FashionMnistDataLoader("test", batchSize, take)
  testDl.zeroCenter(trainDl.meanImage)

  val numFeatures = trainDl.numFeatures
  val numClasses = 10

//  val w = Variable(ns.randn(numFeatures, numClasses))
  val w = Variable(ns.randn(numFeatures, numClasses) * 0.01) // random initialization
//  val w = Variable(ns.randn(numFeatures, numClasses) * math.sqrt(1.0 / numClasses)) // Xavier initialization
//  val w = Variable(ns.randn(numFeatures, numClasses) * math.sqrt(2.0 / numClasses)) // He initialization
  val b = Variable(ns.zeros(numClasses))

  def net(x: Variable): Variable = (x dot w) + b

  def loss(yHat: Variable, y: Variable): Variable = SoftmaxLoss(yHat, y).forward()

  def sgd(params: Seq[Variable], lr: Double, batchSize: Int): Unit =
    params.foreach { p: Variable =>
      p.data -= lr * p.g
      p.g := 0.0
    }

  def evaluate(dl: DataLoader, net: Variable => Variable): (Double, Double) = {
    val (l, a) =
      dl.foldLeft(0.0, 0.0) {
        case ((lossAcc, accuracyAcc), (x, y)) =>
          val output = net(x)
          val guessed = ns.argmax(output.data, axis = 1)
          val accuracy = ns.sum(guessed == y.data)
          val cost = loss(output, y).data.squeeze()
          (lossAcc + cost, accuracyAcc + accuracy)
      }
    (l / dl.numBatches, a / dl.numSamples)
  }

  val numEpochs = 5
  val lr = 0.5

  (0 until numEpochs) foreach { epoch =>
    trainDl.foreach {
      case (x, y) =>
        val yh = net(x)
        val l = loss(yh, y)
        l.backward()
        sgd(Seq(w, b), lr, batchSize) // update parameters using their gradient
    }
    val (ltrn, atrn) = evaluate(trainDl, net)
    val (ltst, atst) = evaluate(testDl, net)

    logger.info(f"epoch $epoch\tloss: $ltst%1.4f/$ltrn%1.4f\taccuracy: $atst%1.4f/$atrn%1.4f")
  }

}
