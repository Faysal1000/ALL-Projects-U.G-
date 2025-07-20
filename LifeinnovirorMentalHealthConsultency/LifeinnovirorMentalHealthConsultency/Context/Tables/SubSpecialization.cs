using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class SubSpecialization
    {
        [Key]
        public int SubSpecializationId { get; set; }

        [Required(ErrorMessage = "Specialization reference is required.")]
        [ForeignKey("Specialization")]
        public int SpecializationId { get; set; }

        public Specialization Specialization { get; set; }

        [Required(ErrorMessage = "Sub specialization name is required.")]
        [StringLength(100, ErrorMessage = "Sub specialization name cannot exceed 100 characters.")]
        public string Name { get; set; }
    }
}