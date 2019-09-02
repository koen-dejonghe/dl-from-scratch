package botkop.linreg

import botkop.{numsca => ns}
import ns._
import scorch.autograd.Variable

import scala.util.Random

object LinearRegression extends App {

  val numFeatures = 2
  val numExamples = 1000

  val trueW = Tensor(2, -3.4).reshape(2, 1)
  val trueB = Tensor(4.2)

  val features = ns.randn(numExamples, numFeatures)

  val labels = (features dot trueW) + trueB
  // noise term epsilon to account for measurement errors
  val epsilon = ns.randn(labels.shape) * 0.01
  labels += epsilon

  def dataIter(batchSize: Int, features: Tensor, labels: Tensor): Iterator[(Variable, Variable)] = {
    val numExamples = features.shape(0)
    // examples are read at random, in no particular order
    val indices = Random.shuffle((0 until numExamples).toList)

    def extract(t: Tensor, indices: Seq[Int]): Variable =
      Variable(ns.concatenate(indices.map(i => t(i))))

    indices.sliding(batchSize, batchSize).map { l =>
      (extract(features, l), extract(labels.T, l))
    }
  }

  val batchSize = 10
//  dataIter(batchSize, features, labels).foreach { case (f, l) =>
//    println(f)
//    println(l)
//    println
//  }

  val w = Variable(ns.randn(numFeatures, 1) * 0.01)
  val b = Variable(ns.zeros(1))

  def linReg(x: Variable, w: Variable, b: Variable): Variable = (x dot w) + b

  def squaredLoss(yHat: Variable, y: Variable): Variable = (yHat - y) ** 2 / 2

  def sgd(params: Seq[Variable], lr: Double, batchSize: Int): Unit =
    params.foreach { p =>
      p.data -= lr * p.grad.data / batchSize
      p.grad.data := 0.0
    }

  val lr = 0.03 // learning rate (step size)
  val numEpochs = 15 // number of iterations
  val net: (Variable, Variable, Variable) => Variable = linReg // our fancy linear model
  val loss: (Variable, Variable) => Variable = squaredLoss // 0.5 (y-y')^2

  (0 until numEpochs).foreach { epoch =>
    /*
      Assuming the number of examples can be divided by the batch size, all
      the examples in the training data set are used once in one epoch
      iteration. The features and tags of mini-batch examples are given by x
      and y respectively
    */
    dataIter(batchSize, features, labels).foreach { case (x, y) =>
        val l = loss(net(x, w, b), y).mean() // minibatch loss in x and y
        l.backward() // compute gradient on l with respect to [w,b]
        sgd(Seq(w, b), lr, batchSize) // update parameters using their gradient
    }
    val l = loss(net(Variable(features), w, b), Variable(labels))
    print(s"epoch: $epoch, loss ${l.mean()}\n")
  }

  println(s"error estimating w: ${trueW - w.data}")
  println(s"error estimating b: ${trueB - b.data}")

}
