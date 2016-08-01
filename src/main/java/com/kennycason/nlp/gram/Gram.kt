package com.kennycason.nlp.gram

/**
 * Created by kenny on 6/14/16.
 */
class Gram<T>(val tokens: MutableList<T>) {
    fun buildToken(delimiter: CharSequence = "_") = tokens.joinToString(delimiter)
}