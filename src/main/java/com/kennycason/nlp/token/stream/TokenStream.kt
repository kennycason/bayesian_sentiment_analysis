package com.kennycason.nlp.token

/**
 * Created by kenny on 6/14/16.
 */
interface TokenStream<out T> : Iterable<T> {
    fun get(index: Int): T
    fun size(): Int
    // from inclusive, to exclusive
    fun window(from: Int, to: Int): List<T>
}
