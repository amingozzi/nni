// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const config = require('./.mocharc.json');
const stripTypesOption = ['--no-strip-types', '--no-experimental-strip-types']
    .find(option => process.allowedNodeEnvironmentFlags.has(option));

if (stripTypesOption) {
    config['node-option'] = [stripTypesOption.slice(2)];
}

module.exports = config;
