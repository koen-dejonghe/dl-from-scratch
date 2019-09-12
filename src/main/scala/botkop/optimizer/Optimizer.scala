package botkop.optimizer

import botkop.autograd.Variable

abstract class Optimizer(parameters: Seq[Variable]) {

  def step(epoch: Int)

  def zeroGrad(): Unit = parameters.foreach(p => p.g := 0)

}
