/// Modified source copyright
// Copyright 2022 Takanori Fujiwara.
// Released under the BSD 3-Clause 'New' or 'Revised' License

/// Original source copyright
// Copyright 2021 Observable, Inc.
// Released under the ISC license.
// https://observablehq.com/@d3/grouped-bar-chart

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const mapValue = (data, a) =>
  typeof a === 'function' ? Object.assign(d3.map(data, a), {
    type: 'function'
  }) : // mapping from data to size
    Array.isArray(a) ? Object.assign([...a], {
      type: 'array'
    }) : // array
      Object.assign(data.map(() => a), {
        type: 'constant'
      }); // constant number

export const barChart = (data, {
  svgId = 'bar-chart',
  x = (d, i) => i, // given d in data, returns the (ordinal) x-value
  y = d => d, // given d in data, returns the (quantitative) y-value
  z = () => 1, // given d in data, returns the (categorical) z-value
  marginTop = 30, // top margin, in pixels
  marginRight = 0, // right margin, in pixels
  marginBottom = 30, // bottom margin, in pixels
  marginLeft = 40, // left margin, in pixels
  width = 640, // outer width, in pixels
  height = 400, // outer height, in pixels
  xDomain, // array of x-values
  xRange = [marginLeft, width - marginRight], // [xmin, xmax]
  xPadding = 0.1, // amount of x-range to reserve to separate groups
  yType = d3.scaleLinear, // type of y-scale
  yDomain, // [ymin, ymax]
  yRange = [height - marginBottom, marginTop], // [ymin, ymax]
  zDomain, // array of z-values
  zPadding = 0.05, // amount of x-range to reserve to separate bars
  xFormat, // a format specifier string for the x-axis
  yFormat, // a format specifier string for the y-axis
  xLabel, // a label for the x-axis
  yLabel, // a label for the y-axis
  colors = d3.schemeTableau10, // array of colors
  showXAxis = true,
  showYAxis = true
} = {}) => {
  // Compute values.
  const X = mapValue(data, x);
  const Y = mapValue(data, y);
  const Z = mapValue(data, z);
  const I = d3.range(X.length);

  // Compute default domains, and unique the x- and z-domains.
  if (xDomain === undefined) xDomain = X;
  if (yDomain === undefined) yDomain = [0, d3.max(Y)];
  if (zDomain === undefined) zDomain = Z;

  // Construct scales, axes, and formats.
  const xScale = d3.scaleBand(xDomain, xRange).paddingInner(xPadding);
  const xzScale = d3.scaleBand(zDomain, [0, xScale.bandwidth()]).padding(zPadding);
  const yScale = yType(yDomain, yRange);
  const zScale = d3.scaleOrdinal(zDomain, colors);
  const xAxis = d3.axisBottom(xScale)
    .tickValues(xScale.domain().filter((d, i) => !(i % Math.ceil(X.length / 10))))
    .tickFormat(xFormat);

  const yAxis = d3.axisLeft(yScale).ticks(Math.max(height / 80, 2), yFormat);

  const svg = d3.create('svg')
    .attr('id', svgId)
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height])
    .attr('style', 'max-width: 100%; height: auto; height: intrinsic;');

  if (showXAxis) {
    const xAxisG = svg.append('g')
      .attr('id', `${svgId}-xaxis`)
      .attr('class', 'axis')
      .attr('transform', `translate(0,${height - marginBottom})`)
      .call(xAxis)
      .call(g => g.append('text')
        .attr('x', marginLeft + (width - marginLeft - marginRight) / 2)
        .attr('y', marginBottom - 3)
        .attr('fill', 'currentColor')
        .attr('text-anchor', 'center')
        .text(xLabel));
  }
  if (showYAxis) {
    svg.append('g')
      .attr('id', `${svgId}-yaxis`)
      .attr('class', 'axis')
      .attr('transform', `translate(${marginLeft},0)`)
      .call(yAxis)
      .call(g => g.append('text')
        .attr('x', -marginLeft)
        .attr('y', 10)
        .attr('fill', 'currentColor')
        .attr('text-anchor', 'start')
        .text(yLabel));
  }

  const bar = svg.append('g')
    .attr('stroke', '#888888')
    .attr('stroke-width', 0.5)
    .selectAll('rect')
    .data(I)
    .join('rect')
    .attr('x', i => xScale(X[i]) + xzScale(Z[i]))
    .attr('y', i => yScale(Y[i]))
    .attr('width', xzScale.bandwidth())
    .attr('height', i => yScale(0) - yScale(Y[i]))
    .attr('fill', i => zScale(Z[i]));

  return svg.node();
}