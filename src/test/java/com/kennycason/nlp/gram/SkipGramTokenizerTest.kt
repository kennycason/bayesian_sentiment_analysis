package com.kennycason.nlp.gram

import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.expect

/**
 * Created by kenny on 6/14/16.
 */
class SkipGramTokenizerTest {
    private val textTokenizer = EnglishTokenizer()

    @Test(expected = IllegalArgumentException::class)
    fun zerogramN0K0() {
        SkipGramTokenizer<String>(0, 0)
    }

    @Test
    fun skipgramsN3K2() {
        val skipgramTokenizer = SkipGramTokenizer<String>(n = 3, k = 2)
        val skipgrams = skipgramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("1 2 3 4 5")))
        assertEquals(10, skipgrams.size)
        assertEquals("1_2_3", skipgrams.get(0).buildToken())
        assertEquals("1_2_4", skipgrams.get(1).buildToken())
        assertEquals("1_2_5", skipgrams.get(2).buildToken())
        assertEquals("1_3_4", skipgrams.get(3).buildToken())
        assertEquals("1_3_5", skipgrams.get(4).buildToken())
        assertEquals("1_4_5", skipgrams.get(5).buildToken())
        assertEquals("2_3_4", skipgrams.get(6).buildToken())
        assertEquals("2_3_5", skipgrams.get(7).buildToken())
        assertEquals("2_4_5", skipgrams.get(8).buildToken())
        assertEquals("3_4_5", skipgrams.get(9).buildToken())
    }

    @Test
    fun skipgramsN3K1() {
        val skipgramTokenizer = SkipGramTokenizer<String>(n = 3, k = 1)
        val skipgrams = skipgramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("1 2 3 4 5")))
        assertEquals(7, skipgrams.size)
        assertEquals("1_2_3", skipgrams.get(0).buildToken())
        assertEquals("1_2_4", skipgrams.get(1).buildToken())
        assertEquals("1_3_4", skipgrams.get(2).buildToken())
        assertEquals("2_3_4", skipgrams.get(3).buildToken())
        assertEquals("2_3_5", skipgrams.get(4).buildToken())
        assertEquals("2_4_5", skipgrams.get(5).buildToken())
        assertEquals("3_4_5", skipgrams.get(6).buildToken())
    }

    @Test
    fun skipgramsN2K1() {
        val skipgramTokenizer = SkipGramTokenizer<String>(n = 2, k = 1)
        val skipgrams = skipgramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("1 2 3 4 5 ")))
        assertEquals(7, skipgrams.size)
        assertEquals("1_2", skipgrams.get(0).buildToken())
        assertEquals("1_3", skipgrams.get(1).buildToken())
        assertEquals("2_3", skipgrams.get(2).buildToken())
        assertEquals("2_4", skipgrams.get(3).buildToken())
        assertEquals("3_4", skipgrams.get(4).buildToken())
        assertEquals("3_5", skipgrams.get(5).buildToken())
        assertEquals("4_5", skipgrams.get(6).buildToken())
    }

    // should function as a bigram tokenizer
    @Test
    fun skipgramsN2K0() {
        val skipgramTokenizer = SkipGramTokenizer<String>(n = 2, k = 0)
        val skipgrams = skipgramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("1 2 3 4 5 ")))
        assertEquals(4, skipgrams.size)
        assertEquals("1_2", skipgrams.get(0).buildToken())
        assertEquals("2_3", skipgrams.get(1).buildToken())
        assertEquals("3_4", skipgrams.get(2).buildToken())
        assertEquals("4_5", skipgrams.get(3).buildToken())
    }
}