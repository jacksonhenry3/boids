struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) particle_pos: vec2<f32>,
    @location(1) particle_vel: vec2<f32>,
    @location(2) particle_color: f32,
};

@vertex
fn main_vs(
    @location(0) particle_pos: vec2<f32>,
    @location(1) particle_vel: vec2<f32>,
    @location(3) position: vec2<f32>,
) -> VertexOutput {
    let angle = -atan2(particle_vel.x, particle_vel.y);
    let pos = vec2<f32>(
        position.x * cos(angle)*0.2 - position.y * sin(angle)*0.2,
        position.x * sin(angle)*0.2 + position.y * cos(angle)*0.2
    );
    let clip_position = vec4<f32>(pos + particle_pos, 0.0, 1.0);
    return VertexOutput(clip_position, vec2<f32>(0.0,0.0), particle_vel,0.0);
}



@fragment
fn main_fs(in:VertexOutput) -> @location(0) vec4<f32> {
    //magnitude of the velocity
    let mag = 1f*sqrt(in.particle_vel.x*in.particle_vel.x + in.particle_vel.y*in.particle_vel.y);
    
    return vec4<f32>(mag, 0.0, 1.0-mag, 0.1);
}