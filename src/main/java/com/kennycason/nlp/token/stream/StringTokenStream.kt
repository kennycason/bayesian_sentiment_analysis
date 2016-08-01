package com.kennycason.nlp.token

import org.eclipse.collections.api.list.ListIterable

/**
 * Created by kenny on 6/14/16.
 *
 * A tokenizer to wrap any default String tokenizer.
 * This abstraction is so that gram tokenizers don't have to care about tokenizer implementations
 */
class StringTokenStream(val tokens: ListIterable<String>) : TokenStream<String> {
    override fun get(index: Int) = tokens.get(index)

    override fun size() = tokens.size()

    override fun window(from: Int, to: Int): List<String> {
        val window = mutableListOf<String>()
        (from..to - 1).forEach { i -> window.add(get(i)) }
        return window
    }

    override fun iterator(): Iterator<String> {
        return object : Iterator<String> {
            val delegateIterator = tokens.iterator()
            override fun hasNext() = delegateIterator.hasNext()
            override fun next() = delegateIterator.next()
        }
    }

}
