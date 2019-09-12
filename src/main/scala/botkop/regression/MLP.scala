package botkop.regression

import java.util.Locale

import botkop.autograd.{SoftmaxLoss, Variable}
import botkop.data.FashionMnistDataLoader
import botkop.numsca.Tensor
import botkop.optimizer.{Adam, SGD}
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging

object MLP extends App with LazyLogging {

  Locale.setDefault(Locale.US)
  val batchSize = 256

  // hyper parameters
  implicit val initMethod: String = "He"
//  val learningRate = 0.5
  val learningRate = 0.01
  val learningRateDecay = 1e-3
  val numHidden = 100

  val numEpochs = 100

//  val take = Some(100)
  val take = None

  logger.info("reading training data set")
  val trainDl = new FashionMnistDataLoader("train", batchSize, take)
  logger.info("reading test data set")
  val testDl = new FashionMnistDataLoader("test", batchSize, take)
  testDl.zeroCenter(trainDl.meanImage)

  val numFeatures = trainDl.numFeatures
  val numClasses = 10

  val w0 = Variable(init(numFeatures, numHidden))
  val b0 = Variable(ns.zeros(numHidden))

  val w1 = Variable(init(numHidden, numClasses))
  val b1 = Variable(ns.zeros(numClasses))

  val parameters = Seq(w0, b0, w1, b1)

  def init(shape: Int*)(implicit method: String): Tensor = {
    ns.randn(shape.toArray) * {
      method match {
        case "Xavier" => math.sqrt(1.0 / shape.last)
        case "He"     => math.sqrt(2.0 / shape.last)
        case _        => 0.01
      }
    }
  }

  def net(x: Variable): Variable = (((x dot w0) + b0).relu() dot w1) + b1

  def loss(yHat: Variable, y: Variable): Variable =
    SoftmaxLoss(yHat, y).forward()

  val sgd = SGD(parameters, learningRate, learningRateDecay)
  val adam = Adam(parameters, learningRate)

  val optimizer = adam

  def evaluate(dl: FashionMnistDataLoader,
               net: Variable => Variable): (Double, Double) = {
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

  (0 until numEpochs) foreach { epoch =>
    val t0 = System.nanoTime()
    trainDl.foreach { // for each mini batch
      case (x, y) =>
        optimizer.zeroGrad()
        val yh = net(x)
        val l = loss(yh, y)
        l.backward()
        optimizer.step(epoch) // update parameters using their gradient
    }
    val t1 = System.nanoTime()
    val dur = (t1 - t0) / 1000000
    val (ltrn, atrn) = evaluate(trainDl, net)
    val (ltst, atst) = evaluate(testDl, net)

    logger.info(
      f"epoch: $epoch%2d duration: $dur%4dms loss: $ltst%1.4f / $ltrn%1.4f\taccuracy: $atst%1.4f / $atrn%1.4f")
  }
}

