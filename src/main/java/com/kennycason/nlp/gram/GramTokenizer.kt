package com.kennycason.nlp.gram

import com.kennycason.nlp.token.TokenStream
import org.eclipse.collections.api.list.ListIterable

/**
 * Created by kenny on 6/14/16.
 */
interface GramTokenizer<T> {
    fun tokenize(tokens: TokenStream<T>) : List<Gram<T>>
}