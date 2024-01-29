import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import { scatterplot } from './scatterplot.js';
import { model, labelToName } from "./main.js";

export const prepareNetworkPlot = (nodes, links, {
    order
} = {}) => {
    d3.select('#network-plot').selectAll('*').remove();
    const nodeDraw = d3.map(nodes, d => [d.x, d.y]);
    const linkDraw = d3.map(links, d => [d.source, d.target]);
    const labels = d3.map(nodes, d => d.label);
    const colors = d3.map(labels, d => d > 0 ? model.labelColors[1] : model.labelColors[d]);
    const width = d3.select('#network-plot').node().getBoundingClientRect().width;
    const network = scatterplot(nodeDraw, {
        svgId: 'network',
        links: linkDraw,
        c: colors,
        order: order,
        width: width,
        height: d3.select('#network-plot').node().getBoundingClientRect().height,
        showXAxis: false,
        showYAxis: false,
        strokeWidth: 0.1,
        stroke: '#b0b0b0',
        linkStrokeWidth: 0.5,
        linkStrokeOpacity: 0.6,
        r: 2.5,
        marginTop: 15,
        marginRight: 20,
        marginBottom: 15,
        marginLeft: 20
    });
    d3.select('#network-plot').append(() => network);

    const networkSvg = d3.select('svg#network');
    networkSvg.append('g')
        .attr('transform', `translate(${width - 80}, 5)`)
        .attr('stroke', '#888888')
        .attr('stroke-width', 0.5)
        .selectAll('circle')
        .data([model.labelColors[0], model.labelColors[1], model.labelColors[-1]])
        .join('circle')
        .attr('fill', d => d)
        .attr('cx', 10)
        .attr('cy', (d, i) => 10 + i * 12)
        .attr('r', 3);
    networkSvg.append('g')
        .attr('transform', `translate(${width - 80}, 5)`)
        .selectAll('text')
        .data([0, 1, -1])
        .join('text')
        .attr('x', 20)
        .attr('y', (d, i) => 10 + i * 12)
        .attr('alignment-baseline', 'middle')
        .attr('fill', '#444444')
        .style('font-size', 10)
        .text(d => labelToName(d))
        .on('mouseover', (event, d, i) => {
            network.update(null, {
                mode: 'selected',
                selected: !network.order ?
                    labels.map(l => l == d) : network.order.map(idx => labels.map(l => l == d)[idx]),
                // selectedColor: model.labelColors[d],
            });
        })
        .on('mouseout', (event, d, i) => {
            console.log(network.order);
            network.update(null, {
                mode: 'default'
            });
        });

    return network;
};
