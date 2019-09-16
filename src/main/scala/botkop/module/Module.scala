package botkop.module

import botkop.autograd.Variable
import botkop.numsca.Tensor

abstract class Module(localParameters: Seq[Variable] = Nil) {

  // by default, obtain submodules through introspection
  lazy val subModules: Seq[Module] =
    this.getClass.getDeclaredFields.flatMap { f =>
      f setAccessible true
      f.get(this) match {
        case module: Module => Some(module)
        case _ => None
      }
    }

  def parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  def gradients: Seq[Tensor] = parameters.map(_.g)

  def zeroGrad(): Unit =
    parameters.foreach(p => p.g := 0)

  /*
    Pytorch way of solving distinction between training and test mode is by using a mutable variable.
    Perhaps there is a better way.
     */
  var isTraining: Boolean = true

  /*
  Sets the module in training mode.
  This has any effect only on modules such as Dropout or BatchNorm.
   */
  def setTrainingMode(mode: Boolean = true): Unit = {
    this.isTraining = mode
    subModules.foreach(_.setTrainingMode(mode))
  }

  def forward(x: Variable): Variable

  def apply(x: Variable): Variable = forward(x)
}

