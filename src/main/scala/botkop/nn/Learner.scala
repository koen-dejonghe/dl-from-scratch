package botkop.nn

import java.util.Locale

import botkop.autograd.Variable
import botkop.data.DataLoader
import botkop.module.Module
import botkop.optimizer.Optimizer
import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging

case class Learner(trainingDataLoader: DataLoader,
                   testDataLoader: DataLoader,
                   net: Module,
                   optimizer: Optimizer,
                   loss: (Variable, Variable) => Variable) extends LazyLogging {

  Locale.setDefault(Locale.US)

  def fit(numEpochs: Int): Unit = {
    (0 until numEpochs) foreach { epoch =>
      val t0 = System.nanoTime()
      trainingDataLoader.foreach { // for each mini batch
        case (x, y) =>
          optimizer.zeroGrad()
          val yh = net(x)
          val l = loss(yh, y)
          l.backward()
          optimizer.step(epoch) // update parameters using their gradient
      }
      val t1 = System.nanoTime()
      val dur = (t1 - t0) / 1000000
      val (ltrn, atrn) = evaluate(trainingDataLoader, net)
      val (ltst, atst) = evaluate(testDataLoader, net)

      logger.info(
        f"epoch: $epoch%2d duration: $dur%4dms loss: $ltst%1.4f / $ltrn%1.4f\taccuracy: $atst%1.4f / $atrn%1.4f")
    }
  }

  def evaluate(dl: DataLoader,
               net: Module): (Double, Double) = {
    net.setTrainingMode(false)
    val (l, a) =
      dl.foldLeft(0.0, 0.0) {
        case ((lossAcc, accuracyAcc), (x, y)) =>
          val output = net(x)
          val guessed = ns.argmax(output.data, axis = 1)
          val accuracy = ns.sum(guessed == y.data)
          val cost = loss(output, y).data.squeeze()
          (lossAcc + cost, accuracyAcc + accuracy)
      }
    net.setTrainingMode()
    (l / dl.numBatches, a / dl.numSamples)
  }
}
