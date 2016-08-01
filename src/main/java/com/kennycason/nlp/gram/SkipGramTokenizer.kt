package com.kennycason.nlp.gram

import com.kennycason.nlp.token.TokenStream
import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists
import org.eclipse.collections.impl.list.mutable.ListAdapter
import java.util.*

/**
 * Created by kenny on 6/14/16.
 *
 * Naive recursive implementation.
 *
 * TODO: implement iterative method
 *       consider bulk skipgram generator from trie
 *       consider consuming List<String> instead of List<OrderedToken>
 *       consider migrating fom String tokens to hashcode tokens
 *
    [1,2,3,4,5]
    n = 3, k = 2 (trigram, skip 2)      n = 3, k = 1 (trigram, skip 3)
    123                                 123
    124                                 124
    125                                 134
    134                                 234
    135                                 235
    145                                 245
    234                                 345
    235
    245
    345
 */

class SkipGramTokenizer<T>(val n: Int, val k: Int) : GramTokenizer<T> {
    init {
        if (n < 1) {
            throw IllegalArgumentException("n must be >= 1")
        }
        if (k < 0) {
            throw IllegalArgumentException("k must be >= 0")
        }
    }

    override fun tokenize(tokens: TokenStream<T>) : List<Gram<T>> {
        if (k == 0 || n < 2) {
            return NGramTokenizer<T>(n).tokenize(tokens);
        }
        if (n == tokens.size()) { return Lists.mutable.empty() }
        val skipgrams = Lists.mutable.empty<Gram<T>>()
        (0..tokens.size() - n).forEach { i ->
            skipgrams.addAll(buildSkipGramsForRange(tokens.window(i, tokens.size()), n, k))
        }
        return skipgrams
    }

    fun buildSkipGramsForRange(tokens: List<T>, n: Int, k: Int): List<Gram<T>> {
        if (n == 1) {
            return Lists.mutable.of(
                    Gram<T>(tokens = mutableListOf(tokens.first())))
        }
        val skipGrams = mutableListOf<Gram<T>>()
        (0..Math.min(k + 1, tokens.size - 1) - 1).forEach { j ->
            val kMinusJSkipMinus1grams = buildSkipGramsForRange(tokens.subList(j + 1, tokens.size), n - 1, k - j)
            for (gram in kMinusJSkipMinus1grams) {
                val skipgram = Gram(tokens = mutableListOf(tokens.first()))
                skipgram.tokens.addAll(gram.tokens)
                skipGrams.add(skipgram)
            }
        }
        return skipGrams
    }
}