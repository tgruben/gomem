package dataframe

import (
	"github.com/apache/arrow/go/v10/arrow"
)

// returns
type ChunkResolver struct {
	offsets     []int
	cachedChunk int
	NumRows     int
}

func NewChunkResolver(chunks *arrow.Column) ChunkResolver {
	numOffsets := len(chunks.Data().Chunks()) + 1
	offsets := make([]int, numOffsets, numOffsets)
	offset := 0
	for i, chunk := range chunks.Data().Chunks() {
		offset += chunk.Len()
		offsets[i+1] = offset
	}
	return ChunkResolver{offsets: offsets, NumRows: offset}
}

func (cr *ChunkResolver) Resolve(idx int) (int, int) {
	if len(cr.offsets) <= 1 {
		return 0, idx
	}
	if idx >= cr.offsets[cr.cachedChunk] && idx < cr.offsets[cr.cachedChunk+1] {
		return cr.cachedChunk, idx - cr.offsets[cr.cachedChunk]
	}
	chunkIndex := cr.Bisect(idx)
	cr.cachedChunk = chunkIndex
	return chunkIndex, idx - cr.offsets[chunkIndex]
}

func (cr *ChunkResolver) Bisect(idx int) int {
	// Search [lo, lo + n)
	lo := 0
	n := len(cr.offsets)
	for n > 1 {
		m := n >> 1
		mid := lo + m
		if idx >= cr.offsets[mid] {
			lo = mid
			n -= m
		} else {
			n = m
		}
	}
	return lo
}
