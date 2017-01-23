package com.kennycason.nlp.gram

import com.kennycason.nlp.token.tokenizer.WhiteSpaceTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import org.eclipse.collections.impl.factory.Lists
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.expect

/**
 * Created by kenny on 1/23/17.
 */
class CompositeGramTokenizerTest {
    private val textTokenizer = EnglishTokenizer()

    @Test(expected = IllegalArgumentException::class)
    fun noTokenizers() {
        CompositeGramTokenizer<String>(Lists.mutable.empty())
    }

    @Test
    fun bigramAndTrigram() {
        val compositeTokenizer = CompositeGramTokenizer<String>(
                Lists.mutable.of(NGramTokenizer<String>(2), NGramTokenizer<String>(3)))

        val allTokens = compositeTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("I really love kotlin")))
        assertEquals(5, allTokens.size)
        assertEquals("i_really", allTokens.get(0).buildToken())
        assertEquals("really_love", allTokens.get(1).buildToken())
        assertEquals("love_kotlin", allTokens.get(2).buildToken())
        assertEquals("i_really_love", allTokens.get(3).buildToken())
        assertEquals("really_love_kotlin", allTokens.get(4).buildToken())
    }

}