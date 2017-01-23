package com.kennycason.nlp.gram

import com.kennycason.nlp.token.TokenStream
import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists

/**
 * Created by kenny on 1/23/17.
 *
 * This class will merge the outuput of multiple tokenizers.
 * This is valuable when you want to train on both bigram and trigrams for example.
 */
class CompositeGramTokenizer<T>(val tokenizers: List<GramTokenizer<T>>) : GramTokenizer<T> {

    init {
        if (tokenizers.isEmpty()) {
            throw IllegalArgumentException("Must pass in at least one tokenizer.")
        }
    }

    override fun tokenize(tokens: TokenStream<T>): List<Gram<T>> {
        val allTokens: MutableList<Gram<T>> = Lists.mutable.empty()
        tokenizers.forEach { tokenizer ->
            allTokens.addAll(tokenizer.tokenize(tokens))
        }
        return allTokens
    }

}