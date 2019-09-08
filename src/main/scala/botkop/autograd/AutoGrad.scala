package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j

object ChainRule1 extends App {
  val x = Variable(-2)
  val y = Variable(5)
  val z = Variable(-4)

  val q = x + y // 3

  val f = q * z // -12

  // print the computation graph
  println(s"computation graph: $f")

  // backprop
  f.backward()

  // print gradients
  println(s"gradient of f = ${f.g}")
  println(s"gradient of q = ${q.g}")
  println(s"gradient of x = ${x.g}")
  println(s"gradient of y = ${y.g}")
  println(s"gradient of z = ${z.g}")
}


