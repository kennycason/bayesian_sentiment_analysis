package com.kennycason.nlp.gram

import com.kennycason.nlp.token.TokenStream
import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists
import java.util.*

/**
 * Created by kenny on 6/14/16.
 */
class NGramTokenizer<T>(val n: Int) : GramTokenizer<T> {
    init {
        if (n < 1) {
            throw IllegalArgumentException("n must be >= 1")
        }
    }

    override fun tokenize(tokens: TokenStream<T>) : List<Gram<T>> {
        val ngrams = Lists.mutable.empty<Gram<T>>()
        for (i in 0..tokens.size() - n) {
            ngrams.add(buildGram(i, i + n - 1, tokens))
        }
        return ngrams
    }

    private fun buildGram(start: Int, end: Int, tokens: TokenStream<T>): Gram<T> {
        val gramTokens = mutableListOf<T>()
        (start..end).forEach { i -> gramTokens.add(tokens.get(i)) }
        return Gram<T>(gramTokens)
    }
}