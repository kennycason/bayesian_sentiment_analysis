package com.kennycason.nlp.token.tokenizer

import org.eclipse.collections.api.list.ListIterable

/**
 * Created by kenny on 7/29/16.
 */
interface Tokenizer<T> {
    fun tokenize(text: String): ListIterable<T>
}