package botkop

object FancyIndexing extends App {

  // val data = List.fill(24)(scala.util.Random.nextDouble())
  val shape = List(2, 3, 4)
  val data = (0 until shape.product).toList.map(_.toDouble)
  val array = Ndarray(data, shape)

  array(0)

  print(array.select(List(0), List(1), List(1)))

}

case class Selection(indices: List[Int], shape: List[Int]) {
//  require(indices.length == shape.product)
}

//object Selection {
//  def apply(ndarray: Ndarray): Selection = {
//    Selection(ndarray.data.indices.toList, ndarray.shape)
//  }
//}

case class Ndarray(data: List[Double],
                   shape: List[Int],
                   selection: Selection) {

  require(data.length == shape.product)

  def apply(ix: Int*): Ndarray = ix.length match {
    case 0 => this
    case _ =>
      val d = data.grouped(shape.tail.product).toList(ix.head)
      Ndarray(d, shape.tail).apply(ix.tail: _*)
  }

  def select(ix: List[Int]*): Ndarray = ix.length match {
    case 0 => this
    case _ =>
      val is = selection.indices.grouped(selection.shape.tail.product).toList
      val si = ix.head.flatMap(i => is(i))
      this.copy(selection = Selection(si, selection.shape.tail)).select(ix.tail: _*)
  }

  def fromSelection: Ndarray = {
    val d = selection.indices.map(data(_))
    Ndarray(d, selection.shape)
  }

}

object Ndarray {
  def apply(data: List[Double], shape: List[Int]): Ndarray = {
    val s = Selection(data.indices.toList, shape)
    Ndarray(data, shape, s)
  }
}

