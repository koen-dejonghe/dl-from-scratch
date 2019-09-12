package botkop.optimizer

import botkop.autograd.Variable


case class SGD(parameters: Seq[Variable],
               learningRate: Double,
               learningRateDecay: Double)
    extends Optimizer(parameters) {

  override def step(epoch: Int): Unit = {
    val alr = learningRate * (1.0 / (1.0 + learningRateDecay * epoch)) // learning rate annealing: 1/t decay
    parameters.foreach { p =>
      p.data -= alr * p.g
    }
  }

}
