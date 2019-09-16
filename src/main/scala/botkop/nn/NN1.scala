package botkop.nn

import java.util.Locale

import botkop.autograd.{SoftmaxLoss, Variable}
import botkop.data.FashionMnistDataLoader
import botkop.module.{Dropout, Linear, Module}
import botkop.optimizer.{Adam, SGD}
import botkop.{numsca => ns}
import botkop.autograd.Function._
import com.typesafe.scalalogging.LazyLogging

object NN1 extends App with LazyLogging {

  Locale.setDefault(Locale.US)
  val batchSize = 1024
  val learningRate = 0.03
  val learningRateDecay = 1e-3
  val numEpochs = 10000

  logger.info("reading training data set")
  val trainDl = new FashionMnistDataLoader("train", batchSize)
  logger.info("reading test data set")
  val testDl = new FashionMnistDataLoader("test", batchSize)
  testDl.zeroCenter(trainDl.meanImage)

  case class Net() extends Module {
    val fc1 = Linear(784, 100)
    val dro = Dropout()
    val fc2 = Linear(100, 10)
    override def forward(x: Variable): Variable = x ~> fc1 ~> dro ~> relu ~> fc2
  }

  val net = Net()

  def loss(yHat: Variable, y: Variable): Variable =
    SoftmaxLoss(yHat, y).forward()

  val sgd = SGD(net.parameters, learningRate, learningRateDecay)
  val adam = Adam(net.parameters, learningRate)

  val optimizer = sgd

  def evaluate(dl: FashionMnistDataLoader,
               net: Module): (Double, Double) = {
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
