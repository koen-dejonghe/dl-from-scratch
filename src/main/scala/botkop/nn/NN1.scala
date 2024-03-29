package botkop.nn

import botkop.autograd.Function._
import botkop.autograd.{SoftmaxLoss, Variable}
import botkop.data.FashionMnistDataLoader
import botkop.module.{BatchNorm, Dropout, Linear, Module}
import botkop.optimizer.{Adam, SGD}
import com.typesafe.scalalogging.LazyLogging

object NN1 extends App with LazyLogging {

  /*

  val batchSize = 256
  val numEpochs = 10000

  val take = None

  logger.info("reading training data set")
  val trainDl = new FashionMnistDataLoader("train", batchSize, take)
  logger.info("reading test data set")
  val testDl = new FashionMnistDataLoader("test", batchSize, take)
  testDl.zeroCenter(trainDl.meanImage)

  case class Net() extends Module {
    val fc1 = Linear(784, 100)
    val dro = Dropout(0.8)
    val fc2 = Linear(100, 10)
    override def forward(x: Variable): Variable = x ~>
      fc1 ~>
      dro ~>
      relu ~>
      fc2
  }

  val nn2: Module = new Module() {
    val fc1 = Linear(784, 250)
    val bn1 = BatchNorm(250)
    val fc2 = Linear(250, 100)
    val bn2 = BatchNorm(100)
    val fc3 = Linear(100, 10)
    val drp = Dropout()
    override def forward(x: Variable): Variable = x ~>
        fc1 ~> bn1 ~> relu ~> drp ~>
        fc2 ~> bn2 ~> relu ~> drp ~>
        fc3
  }

  def loss(yHat: Variable, y: Variable): Variable =
    SoftmaxLoss(yHat, y).forward()

  val sgd = SGD(net.parameters, 0.1, 1e-3)
  val adam = Adam(net.parameters, 1e-3)

//  Learner(trnDl, tstDl, nn1, adam, loss).fit(numEpochs)
  Learner(trnDl, tstDl, nn2, adam, loss).fit(numEpochs)
  */


}
